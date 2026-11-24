"""5.1 Gold-label judge calibration script comparing scoring protocols."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig
from .mqm_protocol import (
    compare_pairwise_once,
    compare_pairwise_swapped,
    hybrid_compare_pair,
    score_direct_total,
    score_pointwise_mqm,
)
from .reward_scoring import count_text_units


@dataclass
class CalibrationMetrics:
    """Agreement, tie rate, position bias, and length bias statistics on gold pairs."""

    protocol: str
    total_pairs: int
    decided_pairs: int
    correct_pairs: int
    agreement_excl_ties: float
    tie_rate: float
    position_bias_rate: float
    length_bias_corr: float


def compute_pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient safely."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    x_std = float(np.std(xa))
    y_std = float(np.std(ya))
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def load_gold_pairs(path: Path) -> list[dict[str, Any]]:
    """Load gold-labeled pairwise comparison records from jsonl."""
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("gold_winner") not in {"A", "B"}:
                continue
            pairs.append(item)
    return pairs


async def calibrate_protocol(
    client: AsyncOpenAIJudgeClient,
    gold_pairs: list[dict[str, Any]],
    protocol: str,
    *,
    diff_threshold: float = 2.0,
) -> CalibrationMetrics:
    """Evaluate a judge protocol against human gold pairwise labels."""
    decided = 0
    correct = 0
    ties = 0
    pos_bias_count = 0
    pos_bias_checked = 0
    len_diffs: list[float] = []
    judge_margin_for_a: list[float] = []

    async def _eval_one(pair: dict[str, Any]):
        src = str(pair["source"])
        cand_a = str(pair["candidate_a"])
        cand_b = str(pair["candidate_b"])
        direction = str(pair.get("direction", "zh2en"))
        gold = str(pair["gold_winner"])
        len_diff = float(count_text_units(cand_a) - count_text_units(cand_b))

        if protocol == "direct_score":
            s_a, s_b = await asyncio.gather(
                score_direct_total(client, src, cand_a, direction),
                score_direct_total(client, src, cand_b, direction),
            )
            winner = "A" if s_a > s_b else ("B" if s_b > s_a else "tie")
            return gold, winner, False, False, len_diff, float(s_a - s_b)

        if protocol == "pointwise_mqm":
            r_a, r_b = await asyncio.gather(
                score_pointwise_mqm(client, src, cand_a, direction),
                score_pointwise_mqm(client, src, cand_b, direction),
            )
            winner = "A" if r_a.total_penalty < r_b.total_penalty else (
                "B" if r_b.total_penalty < r_a.total_penalty else "tie"
            )
            return gold, winner, False, False, len_diff, float(r_b.total_penalty - r_a.total_penalty)

        if protocol == "pairwise_single":
            winner = await compare_pairwise_once(client, src, cand_a, cand_b, direction)
            margin = 1.0 if winner == "A" else (-1.0 if winner == "B" else 0.0)
            return gold, winner, False, False, len_diff, margin

        if protocol == "hybrid_mqm_pairwise":
            hyb = await hybrid_compare_pair(
                client,
                src,
                cand_a,
                cand_b,
                direction,
                diff_threshold=diff_threshold,
            )
            checked = hyb.pairwise_detail is not None
            bias = bool(hyb.pairwise_detail.position_bias_triggered) if hyb.pairwise_detail else False
            margin = float(hyb.penalty_b - hyb.penalty_a)
            if hyb.winner == "A":
                margin = max(margin, 0.5)
            elif hyb.winner == "B":
                margin = min(margin, -0.5)
            return gold, hyb.winner, checked, bias, len_diff, margin

        if protocol == "pairwise_swapped":
            pw = await compare_pairwise_swapped(client, src, cand_a, cand_b, direction)
            margin = 1.0 if pw.winner == "A" else (-1.0 if pw.winner == "B" else 0.0)
            return gold, pw.winner, True, pw.position_bias_triggered, len_diff, margin

        raise ValueError(f"Unknown protocol: {protocol}")

    results = await asyncio.gather(*[_eval_one(p) for p in gold_pairs])
    for gold, winner, checked, bias, len_diff, margin in results:
        len_diffs.append(len_diff)
        judge_margin_for_a.append(margin)
        if checked:
            pos_bias_checked += 1
            if bias:
                pos_bias_count += 1
        if winner == "tie":
            ties += 1
        else:
            decided += 1
            if winner == gold:
                correct += 1

    total = len(gold_pairs)
    return CalibrationMetrics(
        protocol=protocol,
        total_pairs=total,
        decided_pairs=decided,
        correct_pairs=correct,
        agreement_excl_ties=float(correct / decided) if decided > 0 else 0.0,
        tie_rate=float(ties / total) if total > 0 else 0.0,
        position_bias_rate=float(pos_bias_count / pos_bias_checked) if pos_bias_checked > 0 else 0.0,
        length_bias_corr=compute_pearson_r(len_diffs, judge_margin_for_a),
    )


async def run_calibration_suite(
    client: AsyncOpenAIJudgeClient,
    gold_pairs: list[dict[str, Any]],
    protocols: list[str] | None = None,
    diff_threshold: float = 2.0,
) -> list[CalibrationMetrics]:
    """Run calibration across all requested judge protocols."""
    proto_list = protocols or ["direct_score", "pointwise_mqm", "pairwise_single", "hybrid_mqm_pairwise"]
    results = []
    for proto in proto_list:
        results.append(await calibrate_protocol(client, gold_pairs, proto, diff_threshold=diff_threshold))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate LLM judge protocols on gold translation labels.")
    parser.add_argument("--gold-file", required=True, help="Path to gold pairwise jsonl.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--diff-threshold", type=float, default=2.0)
    parser.add_argument("--cache-path", default=None)
    args = parser.parse_args()

    gold_pairs = load_gold_pairs(Path(args.gold_file))
    client = AsyncOpenAIJudgeClient(
        JudgeClientConfig(
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            cache_path=args.cache_path,
        )
    )

    async def _run():
        try:
            metrics_list = await run_calibration_suite(client, gold_pairs, diff_threshold=args.diff_threshold)
            print(json.dumps([m.__dict__ for m in metrics_list], ensure_ascii=False, indent=2))
        finally:
            await client.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
