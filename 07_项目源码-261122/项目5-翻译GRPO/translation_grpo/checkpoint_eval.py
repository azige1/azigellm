"""5.5 Online-judge vs external strong-model consistency diagnostic and checkpoint selection."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import compute_pearson_r
from .judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig
from .mqm_protocol import hybrid_compare_pair, score_pointwise_mqm
from .reward_scoring import count_text_units


@dataclass
class ConsistencyDiagnosticResult:
    """Agreement between online judge and external strong judge over rollout groups."""

    num_groups: int
    num_candidates: int
    penalty_pearson_r: float
    top1_disagreement_rate: float
    disagreement_by_bucket: dict[str, dict[str, float]]


@dataclass
class CheckpointEvalRecord:
    """Evaluation summary for a single checkpoint judged by external strong model."""

    step: int
    checkpoint_path: str
    online_reward_mean: float
    strong_mqm_penalty_mean: float
    pairwise_wins: int
    pairwise_losses: int
    pairwise_ties: int
    pairwise_win_rate_excl_ties: float
    sign_test_z: float
    sign_test_p_value: float


def compute_sign_test(wins: int, losses: int) -> tuple[float, float]:
    """Compute normal-approximated two-sided sign test z-score and p-value."""
    total = wins + losses
    if total <= 0:
        return 0.0, 1.0
    expected = total * 0.5
    std = math.sqrt(total * 0.25)
    z = (wins - expected) / std
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p_value)


async def run_consistency_diagnostic(
    online_client: AsyncOpenAIJudgeClient,
    strong_client: AsyncOpenAIJudgeClient,
    rollout_groups: list[dict[str, Any]],
) -> ConsistencyDiagnosticResult:
    """Compare online judge vs external strong judge across rollout groups.

    Each element of rollout_groups has:
        {"source": str, "direction": str, "candidates": list[str], "has_terminology": bool (optional)}
    """

    async def _eval_group(group: dict[str, Any]):
        src = str(group["source"])
        direction = str(group.get("direction", "zh2en"))
        cands = [str(c) for c in group["candidates"]]
        online_res, strong_res = await asyncio.gather(
            asyncio.gather(*[score_pointwise_mqm(online_client, src, c, direction) for c in cands]),
            asyncio.gather(*[score_pointwise_mqm(strong_client, src, c, direction) for c in cands]),
        )
        online_penalties = [r.total_penalty for r in online_res]
        strong_penalties = [r.total_penalty for r in strong_res]
        online_best = int(np.argmin(online_penalties))
        strong_best = int(np.argmin(strong_penalties))
        disagree = online_best != strong_best
        src_units = count_text_units(src)
        bucket = "long_sentence" if src_units >= 35 else ("terminology" if group.get("has_terminology") else "standard")
        return online_penalties, strong_penalties, disagree, bucket

    group_outputs = await asyncio.gather(*[_eval_group(g) for g in rollout_groups])

    all_online: list[float] = []
    all_strong: list[float] = []
    disagree_count = 0
    bucket_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [disagree, total]

    for o_pens, s_pens, disagree, bucket in group_outputs:
        all_online.extend(o_pens)
        all_strong.extend(s_pens)
        bucket_counts[bucket][1] += 1
        if disagree:
            disagree_count += 1
            bucket_counts[bucket][0] += 1

    n_groups = len(rollout_groups)
    bucket_summary = {
        b: {
            "groups": float(tot),
            "disagree": float(dis),
            "disagreement_rate": float(dis / tot) if tot > 0 else 0.0,
        }
        for b, (dis, tot) in bucket_counts.items()
    }

    return ConsistencyDiagnosticResult(
        num_groups=n_groups,
        num_candidates=len(all_online),
        penalty_pearson_r=compute_pearson_r(all_online, all_strong),
        top1_disagreement_rate=float(disagree_count / n_groups) if n_groups > 0 else 0.0,
        disagreement_by_bucket=bucket_summary,
    )


def detect_overoptimization_step(
    records: list[CheckpointEvalRecord],
    *,
    consecutive_divergences: int = 2,
) -> int | None:
    """Detect when online reward keeps rising while strong-model penalty worsens consecutively."""
    ordered = sorted(records, key=lambda r: r.step)
    streak = 0
    for idx in range(1, len(ordered)):
        prev_r, curr_r = ordered[idx - 1], ordered[idx]
        reward_up = curr_r.online_reward_mean > prev_r.online_reward_mean
        penalty_worsened = curr_r.strong_mqm_penalty_mean > prev_r.strong_mqm_penalty_mean
        if reward_up and penalty_worsened:
            streak += 1
            if streak >= consecutive_divergences:
                return curr_r.step
        else:
            streak = 0
    return None


def select_best_checkpoint(records: list[CheckpointEvalRecord]) -> CheckpointEvalRecord:
    """Select checkpoint solely by external strong judge metrics (never training reward)."""
    if not records:
        raise ValueError("No checkpoint evaluation records provided.")
    return min(
        records,
        key=lambda r: (round(r.strong_mqm_penalty_mean, 6), -r.pairwise_win_rate_excl_ties, r.step),
    )


async def evaluate_checkpoint_vs_baseline(
    strong_client: AsyncOpenAIJudgeClient,
    *,
    step: int,
    checkpoint_path: str,
    eval_samples: list[dict[str, Any]],
    sft_baseline_outputs: list[str],
    checkpoint_outputs: list[str],
    online_reward_mean: float = 0.0,
    diff_threshold: float = 2.0,
) -> CheckpointEvalRecord:
    """Evaluate a checkpoint against SFT baseline with external strong model MQM + Pairwise."""
    if not (len(eval_samples) == len(sft_baseline_outputs) == len(checkpoint_outputs)):
        raise ValueError("eval_samples, sft_baseline_outputs, and checkpoint_outputs must have equal length.")

    async def _score_sample(sample: dict[str, Any], sft_out: str, ckpt_out: str):
        src = str(sample["source"])
        direction = str(sample.get("direction", "zh2en"))
        ckpt_mqm, sft_mqm = await asyncio.gather(
            score_pointwise_mqm(strong_client, src, ckpt_out, direction),
            score_pointwise_mqm(strong_client, src, sft_out, direction),
        )
        hyb = await hybrid_compare_pair(
            strong_client,
            src,
            ckpt_out,
            sft_out,
            direction,
            penalty_a=ckpt_mqm.total_penalty,
            penalty_b=sft_mqm.total_penalty,
            diff_threshold=diff_threshold,
        )
        return ckpt_mqm.total_penalty, hyb.winner

    results = await asyncio.gather(
        *[_score_sample(s, b, c) for s, b, c in zip(eval_samples, sft_baseline_outputs, checkpoint_outputs, strict=True)]
    )

    penalties = [p for p, _ in results]
    wins = sum(1 for _, w in results if w == "A")
    losses = sum(1 for _, w in results if w == "B")
    ties = sum(1 for _, w in results if w == "tie")
    decided = wins + losses
    win_rate = float(wins / decided) if decided > 0 else 0.5
    z, p_val = compute_sign_test(wins, losses)

    return CheckpointEvalRecord(
        step=step,
        checkpoint_path=checkpoint_path,
        online_reward_mean=float(online_reward_mean),
        strong_mqm_penalty_mean=float(sum(penalties) / max(1, len(penalties))),
        pairwise_wins=wins,
        pairwise_losses=losses,
        pairwise_ties=ties,
        pairwise_win_rate_excl_ties=win_rate,
        sign_test_z=z,
        sign_test_p_value=p_val,
    )


async def generate_from_openai_endpoint(
    client: AsyncOpenAIJudgeClient,
    eval_samples: list[dict[str, Any]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> list[str]:
    """Generate translations for eval_samples at temperature 0.7 via an OpenAI-compatible server."""

    async def _gen(sample: dict[str, Any]) -> str:
        direction = str(sample.get("direction", "zh2en"))
        src = str(sample["source"])
        prefix = (
            "Translate the following Chinese text into English. Output only the translation:\n"
            if direction == "zh2en"
            else "将下面的英文翻译为中文，只输出译文：\n"
        )
        return await client.chat(prefix + src, temperature=temperature, max_tokens=max_tokens)

    return list(await asyncio.gather(*[_gen(s) for s in eval_samples]))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="5.5 External strong-model checkpoint selection & consistency check.")
    parser.add_argument("--mode", choices=["consistency", "select"], required=True)
    parser.add_argument("--input-file", required=True, help="JSONL of rollout groups or checkpoint eval manifest.")
    parser.add_argument("--strong-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--strong-model", default="gpt-5")
    parser.add_argument("--strong-api-key-env", default="STRONG_JUDGE_API_KEY")
    parser.add_argument("--online-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--online-model", default="Qwen3-32B")
    parser.add_argument("--online-api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--cache-dir", default="outputs/judge_eval_cache")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    strong_client = AsyncOpenAIJudgeClient(
        JudgeClientConfig(
            base_url=args.strong_base_url,
            model=args.strong_model,
            api_key_env=args.strong_api_key_env,
            cache_path=str(cache_dir / "strong_judge_cache.jsonl"),
        )
    )
    online_client = AsyncOpenAIJudgeClient(
        JudgeClientConfig(
            base_url=args.online_base_url,
            model=args.online_model,
            api_key_env=args.online_api_key_env,
            cache_path=str(cache_dir / "online_judge_cache.jsonl"),
        )
    )

    async def _run():
        try:
            if args.mode == "consistency":
                groups = load_jsonl(Path(args.input_file))
                res = await run_consistency_diagnostic(online_client, strong_client, groups)
                print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))
            else:
                manifest = load_jsonl(Path(args.input_file))
                records: list[CheckpointEvalRecord] = []
                for entry in manifest:
                    eval_samples = load_jsonl(Path(entry["eval_samples_path"]))
                    sft_outputs = [str(r["output"]) for r in load_jsonl(Path(entry["sft_outputs_path"]))]
                    ckpt_outputs = [str(r["output"]) for r in load_jsonl(Path(entry["ckpt_outputs_path"]))]
                    rec = await evaluate_checkpoint_vs_baseline(
                        strong_client,
                        step=int(entry["step"]),
                        checkpoint_path=str(entry["checkpoint_path"]),
                        eval_samples=eval_samples,
                        sft_baseline_outputs=sft_outputs,
                        checkpoint_outputs=ckpt_outputs,
                        online_reward_mean=float(entry.get("online_reward_mean", 0.0)),
                    )
                    records.append(rec)

                best = select_best_checkpoint(records)
                overopt_step = detect_overoptimization_step(records)
                print(
                    json.dumps(
                        {
                            "selected_checkpoint": best.__dict__,
                            "overoptimization_detected_at_step": overopt_step,
                            "all_checkpoints": [r.__dict__ for r in sorted(records, key=lambda x: x.step)],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        finally:
            await asyncio.gather(strong_client.close(), online_client.close())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
