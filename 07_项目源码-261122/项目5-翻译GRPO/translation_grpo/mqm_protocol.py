"""5.1 Hybrid Pointwise MQM + position-swapped Pairwise judge protocol."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .judge_client import (
    DIRECT_SCORE_PROMPT_TEMPLATE,
    PAIRWISE_JUDGE_PROMPT_TEMPLATE,
    POINTWISE_MQM_PROMPT_TEMPLATE,
    AsyncOpenAIJudgeClient,
    render_prompt_template,
)

MQM_DIMENSIONS: tuple[str, ...] = (
    "accuracy",
    "terminology",
    "fluency",
    "style",
    "context",
)

SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 25.0,
    "major": 5.0,
    "minor": 1.0,
}

_DIM_ALIASES: dict[str, str] = {
    "accuracy": "accuracy",
    "准确": "accuracy",
    "准确性": "accuracy",
    "mistranslation": "accuracy",
    "omission": "accuracy",
    "addition": "accuracy",
    "terminology": "terminology",
    "term": "terminology",
    "术语": "terminology",
    "术语性": "terminology",
    "fluency": "fluency",
    "流畅": "fluency",
    "流畅度": "fluency",
    "grammar": "fluency",
    "style": "style",
    "风格": "style",
    "语域": "style",
    "context": "context",
    "locale": "context",
    "localization": "context",
    "本地化": "context",
    "语境": "context",
    "语境/本地化": "context",
}

_SEVERITY_ALIASES: dict[str, str] = {
    "critical": "critical",
    "severe": "critical",
    "严重": "critical",
    "major": "major",
    "主要": "major",
    "minor": "minor",
    "轻微": "minor",
}


def normalize_dimension(dim: str) -> str:
    """Normalize an MQM dimension label into one of the five canonical keys."""
    key = (dim or "").strip().lower()
    if key in _DIM_ALIASES:
        return _DIM_ALIASES[key]
    for alias, canonical in _DIM_ALIASES.items():
        if alias in key:
            return canonical
    return "accuracy"


def normalize_severity(severity: str) -> str:
    """Normalize a severity label into critical, major, or minor."""
    key = (severity or "").strip().lower()
    if key in _SEVERITY_ALIASES:
        return _SEVERITY_ALIASES[key]
    for alias, canonical in _SEVERITY_ALIASES.items():
        if alias in key:
            return canonical
    return "minor"


@dataclass
class MQMErrorSpan:
    """Single annotated MQM error span."""

    span: str
    dim: str
    severity: str
    weight: float


@dataclass
class MQMPointwiseResult:
    """Structured output of Pointwise MQM evaluation."""

    errors: list[MQMErrorSpan]
    dim_penalties: dict[str, float]
    total_penalty: float
    raw_response: str = ""


@dataclass
class PairwiseResult:
    """Output of position-swapped Pairwise comparison."""

    winner: str  # "A", "B", or "tie"
    order_ab_verdict: str
    order_ba_verdict_mapped: str
    consistent: bool
    position_bias_triggered: bool


@dataclass
class HybridPairwiseResult:
    """Output of hybrid Pointwise MQM + Pairwise comparison."""

    winner: str  # "A", "B", or "tie"
    decided_by: str  # "pointwise" or "pairwise"
    penalty_a: float
    penalty_b: float
    pairwise_detail: PairwiseResult | None = None


def parse_mqm_errors(raw_errors: Any) -> list[MQMErrorSpan]:
    """Parse raw JSON error list into normalized MQMErrorSpan objects."""
    if not isinstance(raw_errors, list):
        return []
    parsed: list[MQMErrorSpan] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span", "")).strip()
        dim = normalize_dimension(str(item.get("dim", item.get("dimension", "accuracy"))))
        severity = normalize_severity(str(item.get("severity", "minor")))
        weight = SEVERITY_WEIGHTS[severity]
        parsed.append(MQMErrorSpan(span=span, dim=dim, severity=severity, weight=weight))
    return parsed


def compute_mqm_penalty(errors: list[MQMErrorSpan]) -> tuple[float, dict[str, float]]:
    """Aggregate per-dimension and total MQM penalty programmatically (25 / 5 / 1)."""
    dim_penalties = {dim: 0.0 for dim in MQM_DIMENSIONS}
    for err in errors:
        dim_penalties[err.dim] = dim_penalties.get(err.dim, 0.0) + float(err.weight)
    total_penalty = float(sum(dim_penalties.values()))
    return total_penalty, dim_penalties


def normalize_pairwise_verdict(raw_winner: str) -> str:
    """Normalize a pairwise judge choice to 'A', 'B', or 'tie'."""
    val = (raw_winner or "").strip().upper()
    if val in {"A", "TRANSLATION A", "CANDIDATE A", "FIRST", "1"}:
        return "A"
    if val in {"B", "TRANSLATION B", "CANDIDATE B", "SECOND", "2"}:
        return "B"
    return "tie"


def _flip_verdict(verdict: str) -> str:
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return "tie"


def is_valid_mqm_output(parsed: Any) -> bool:
    """MQM 输出必须是含 ``errors`` 列表的 JSON 对象；空列表表示没有错误，缺字段或类型错误视为无效。"""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("errors"), list):
        return False
    return all(isinstance(item, dict) for item in parsed["errors"])


async def score_pointwise_mqm(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidate: str,
    direction: str = "zh2en",
) -> MQMPointwiseResult:
    """Score a single translation via Pointwise MQM error annotation."""
    prompt = render_prompt_template(
        POINTWISE_MQM_PROMPT_TEMPLATE,
        {
            "__DIRECTION__": direction,
            "__SOURCE__": source,
            "__CANDIDATE__": candidate,
        },
    )
    parsed, raw = await client.chat_json(prompt, validator=is_valid_mqm_output)
    errors = parse_mqm_errors(parsed["errors"])
    total_penalty, dim_penalties = compute_mqm_penalty(errors)
    return MQMPointwiseResult(
        errors=errors,
        dim_penalties=dim_penalties,
        total_penalty=total_penalty,
        raw_response=raw,
    )


async def score_direct_total(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidate: str,
    direction: str = "zh2en",
) -> float:
    """Baseline 0-100 direct scoring protocol (for calibration/ablation comparison)."""
    prompt = render_prompt_template(
        DIRECT_SCORE_PROMPT_TEMPLATE,
        {
            "__DIRECTION__": direction,
            "__SOURCE__": source,
            "__CANDIDATE__": candidate,
        },
    )
    parsed, _ = await client.chat_json(prompt)
    try:
        score = float(parsed.get("score", 50.0))
    except (TypeError, ValueError):
        score = 50.0
    return max(0.0, min(100.0, score))


async def compare_pairwise_once(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidate_a: str,
    candidate_b: str,
    direction: str = "zh2en",
) -> str:
    """Run single-order Pairwise comparison returning 'A', 'B', or 'tie'."""
    prompt = render_prompt_template(
        PAIRWISE_JUDGE_PROMPT_TEMPLATE,
        {
            "__DIRECTION__": direction,
            "__SOURCE__": source,
            "__CANDIDATE_A__": candidate_a,
            "__CANDIDATE_B__": candidate_b,
        },
    )
    parsed, _ = await client.chat_json(prompt)
    return normalize_pairwise_verdict(str(parsed.get("winner", "tie")))


async def compare_pairwise_swapped(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidate_a: str,
    candidate_b: str,
    direction: str = "zh2en",
) -> PairwiseResult:
    """Run position-swapped Pairwise comparison; conflicting orders yield a tie."""
    verdict_ab, verdict_ba_raw = await asyncio.gather(
        compare_pairwise_once(client, source, candidate_a, candidate_b, direction),
        compare_pairwise_once(client, source, candidate_b, candidate_a, direction),
    )
    verdict_ba_mapped = _flip_verdict(verdict_ba_raw)
    consistent = verdict_ab == verdict_ba_mapped
    position_bias_triggered = (verdict_ab == "A" and verdict_ba_raw == "A") or (
        verdict_ab == "B" and verdict_ba_raw == "B"
    )
    if consistent and verdict_ab in {"A", "B"}:
        winner = verdict_ab
    else:
        winner = "tie"
    return PairwiseResult(
        winner=winner,
        order_ab_verdict=verdict_ab,
        order_ba_verdict_mapped=verdict_ba_mapped,
        consistent=consistent,
        position_bias_triggered=position_bias_triggered,
    )


async def hybrid_compare_pair(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidate_a: str,
    candidate_b: str,
    direction: str = "zh2en",
    *,
    penalty_a: float | None = None,
    penalty_b: float | None = None,
    diff_threshold: float = 2.0,
) -> HybridPairwiseResult:
    """Hybrid protocol: Pointwise MQM first; escalate to swapped Pairwise when |dP| < 2."""
    if penalty_a is None or penalty_b is None:
        res_a, res_b = await asyncio.gather(
            score_pointwise_mqm(client, source, candidate_a, direction)
            if penalty_a is None
            else asyncio.sleep(0, result=None),
            score_pointwise_mqm(client, source, candidate_b, direction)
            if penalty_b is None
            else asyncio.sleep(0, result=None),
        )
        if penalty_a is None and res_a is not None:
            penalty_a = res_a.total_penalty
        if penalty_b is None and res_b is not None:
            penalty_b = res_b.total_penalty

    assert penalty_a is not None and penalty_b is not None
    if abs(penalty_a - penalty_b) >= diff_threshold:
        winner = "A" if penalty_a < penalty_b else "B"
        return HybridPairwiseResult(
            winner=winner,
            decided_by="pointwise",
            penalty_a=float(penalty_a),
            penalty_b=float(penalty_b),
            pairwise_detail=None,
        )

    pw = await compare_pairwise_swapped(client, source, candidate_a, candidate_b, direction)
    return HybridPairwiseResult(
        winner=pw.winner,
        decided_by="pairwise",
        penalty_a=float(penalty_a),
        penalty_b=float(penalty_b),
        pairwise_detail=pw,
    )


@dataclass
class GroupHybridScoringResult:
    """Result of scoring 8 candidates + 1 anonymous reference anchor in shuffled order."""

    candidate_results: list[MQMPointwiseResult]
    anchor_result: MQMPointwiseResult | None
    shuffled_order: list[int]
    pairwise_tiebreak_deltas: list[float] = field(default_factory=list)
    pairwise_calls: int = 0


async def score_group_with_anchor_hybrid(
    client: AsyncOpenAIJudgeClient,
    source: str,
    candidates: list[str],
    direction: str = "zh2en",
    *,
    reference_anchor: str | None = None,
    enable_pairwise_tiebreak: bool = True,
    diff_threshold: float = 2.0,
    pairwise_tiebreak_weight: float = 0.25,
    seed: int | None = None,
) -> GroupHybridScoringResult:
    """Mix 1 anonymous reference anchor with candidates, shuffle, and run hybrid MQM+Pairwise."""
    all_texts = list(candidates)
    has_anchor = reference_anchor is not None
    if has_anchor:
        all_texts.append(reference_anchor)

    indices = list(range(len(all_texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    shuffled_tasks = [score_pointwise_mqm(client, source, all_texts[idx], direction) for idx in indices]
    shuffled_results = await asyncio.gather(*shuffled_tasks)

    unshuffled: list[MQMPointwiseResult | None] = [None] * len(all_texts)
    for pos, orig_idx in enumerate(indices):
        unshuffled[orig_idx] = shuffled_results[pos]

    assert all(r is not None for r in unshuffled)
    ordered_results = [r for r in unshuffled if r is not None]

    n_cand = len(candidates)
    candidate_results = ordered_results[:n_cand]
    anchor_result = ordered_results[n_cand] if has_anchor else None

    tiebreak_deltas = [0.0] * n_cand
    pairwise_calls = 0

    if enable_pairwise_tiebreak and n_cand > 1:
        pair_indices: list[tuple[int, int]] = []
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if abs(candidate_results[i].total_penalty - candidate_results[j].total_penalty) < diff_threshold:
                    pair_indices.append((i, j))

        if pair_indices:
            pw_results = await asyncio.gather(
                *[
                    compare_pairwise_swapped(client, source, candidates[i], candidates[j], direction)
                    for i, j in pair_indices
                ]
            )
            pairwise_calls = len(pw_results) * 2
            win_balance = [0.0] * n_cand
            counts = [0] * n_cand
            for (i, j), pw in zip(pair_indices, pw_results, strict=True):
                counts[i] += 1
                counts[j] += 1
                if pw.winner == "A":
                    win_balance[i] += 1.0
                    win_balance[j] -= 1.0
                elif pw.winner == "B":
                    win_balance[i] -= 1.0
                    win_balance[j] += 1.0
            for i in range(n_cand):
                if counts[i] > 0:
                    tiebreak_deltas[i] = pairwise_tiebreak_weight * (win_balance[i] / counts[i])

    return GroupHybridScoringResult(
        candidate_results=candidate_results,
        anchor_result=anchor_result,
        shuffled_order=indices,
        pairwise_tiebreak_deltas=tiebreak_deltas,
        pairwise_calls=pairwise_calls,
    )
