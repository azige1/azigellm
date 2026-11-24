"""5.2 Seven-dimension MQM + repetition/length reward and 5.4 invalid sample detection."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig, JudgeOutputError
from .mqm_protocol import MQM_DIMENSIONS, MQMPointwiseResult, score_direct_total, score_pointwise_mqm

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_UNIT_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


@dataclass
class RewardWeightsConfig:
    """Weights and thresholds for 5.2 7-dim reward and 5.4 invalid detection."""

    accuracy: float = 1.2
    terminology: float = 1.5
    fluency: float = 1.0
    style: float = 1.0
    context: float = 1.0
    lambda_rep: float = 10.0
    lambda_len: float = 5.0
    len_penalty_ratio_threshold: float = 1.5
    short_source_unit_threshold: int = 10
    rep_ngram_n: int = 4
    rep_rate_threshold: float = 0.3
    hard_len_ratio_threshold: float = 2.5
    max_response_tokens: int = 1024
    reward_mode: str = "seven_dim"  # "seven_dim" | "total_score"

    def dimension_weights(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "terminology": self.terminology,
            "fluency": self.fluency,
            "style": self.style,
            "context": self.context,
        }


def tokenize_for_ngram(text: str) -> list[str]:
    """Tokenize bilingual text into CJK characters and lowercased alphanumeric words."""
    if not text:
        return []
    return [tok.lower() for tok in _UNIT_TOKEN_RE.findall(text)]


def count_text_units(text: str) -> int:
    """Count linguistic units (CJK characters + alphanumeric words) in a sentence."""
    return len(tokenize_for_ngram(text))


def compute_ngram_repetition_rate(text: str, n: int = 4) -> float:
    """Compute repeated n-gram ratio = (total_ngrams - unique_ngrams) / total_ngrams."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    tokens = tokenize_for_ngram(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    if total == 0:
        return 0.0
    unique = len(set(ngrams))
    return float((total - unique) / total)


def detect_invalid_output(
    candidate: str,
    source: str,
    *,
    source_units: int | None = None,
    response_token_len: int | None = None,
    config: RewardWeightsConfig | None = None,
) -> dict[str, Any]:
    """Detect repetition and overlength failures for 5.2 penalty and 5.4 hard mask."""
    cfg = config or RewardWeightsConfig()
    rep_rate = compute_ngram_repetition_rate(candidate, n=cfg.rep_ngram_n)
    rep_flag = rep_rate > cfg.rep_rate_threshold

    src_len = int(source_units) if source_units is not None and source_units > 0 else max(1, count_text_units(source))
    cand_len = count_text_units(candidate)
    len_ratio = float(cand_len / src_len)

    hit_len_ratio = len_ratio > cfg.hard_len_ratio_threshold
    hit_max_tokens = response_token_len is not None and int(response_token_len) >= cfg.max_response_tokens
    is_invalid = bool(rep_flag or hit_len_ratio or hit_max_tokens)

    return {
        "rep_rate": rep_rate,
        "rep_flag": rep_flag,
        "source_units": src_len,
        "candidate_units": cand_len,
        "len_ratio": len_ratio,
        "hit_len_ratio": hit_len_ratio,
        "hit_max_tokens": hit_max_tokens,
        "is_invalid": is_invalid,
    }


def compute_seven_dim_reward(
    dim_penalties: dict[str, float],
    *,
    rep_flag: bool,
    len_ratio: float,
    source_units: int,
    config: RewardWeightsConfig | None = None,
) -> dict[str, float]:
    """Compute the 7-dimension translation reward and return per-dimension breakdown."""
    cfg = config or RewardWeightsConfig()
    weights = cfg.dimension_weights()

    weighted_mqm = 0.0
    breakdown: dict[str, float] = {}
    raw_total_penalty = 0.0
    for dim in MQM_DIMENSIONS:
        p_k = float(dim_penalties.get(dim, 0.0))
        raw_total_penalty += p_k
        w_k = float(weights[dim])
        weighted_mqm += w_k * p_k
        breakdown[f"{dim}_penalty"] = p_k
        breakdown[f"{dim}_weighted_penalty"] = w_k * p_k

    rep_penalty = cfg.lambda_rep if rep_flag else 0.0
    if source_units < cfg.short_source_unit_threshold:
        len_penalty = 0.0
    else:
        len_penalty = cfg.lambda_len * max(0.0, float(len_ratio) - cfg.len_penalty_ratio_threshold)

    reward = -(weighted_mqm + rep_penalty + len_penalty)
    breakdown["mqm_penalty"] = raw_total_penalty
    breakdown["weighted_mqm_penalty"] = weighted_mqm
    breakdown["rep_penalty"] = float(rep_penalty)
    breakdown["len_penalty"] = float(len_penalty)
    breakdown["score"] = float(reward)
    return breakdown


def score_candidate_from_pointwise(
    mqm_result: MQMPointwiseResult,
    source: str,
    candidate: str,
    *,
    source_units: int | None = None,
    response_token_len: int | None = None,
    config: RewardWeightsConfig | None = None,
) -> dict[str, Any]:
    """Combine a Pointwise MQM annotation with repetition and length penalties."""
    cfg = config or RewardWeightsConfig()
    flags = detect_invalid_output(
        candidate,
        source,
        source_units=source_units,
        response_token_len=response_token_len,
        config=cfg,
    )
    breakdown = compute_seven_dim_reward(
        mqm_result.dim_penalties,
        rep_flag=flags["rep_flag"],
        len_ratio=flags["len_ratio"],
        source_units=flags["source_units"],
        config=cfg,
    )
    return {
        **breakdown,
        **flags,
        "mqm_errors": [
            {"span": e.span, "dim": e.dim, "severity": e.severity, "weight": e.weight}
            for e in mqm_result.errors
        ],
    }


_SHARED_CLIENT_LOCK = asyncio.Lock() if hasattr(asyncio, "Lock") else None
_SHARED_CLIENTS: dict[str, AsyncOpenAIJudgeClient] = {}


def load_sidecar_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load sidecar translation_grpo.yaml when present."""
    candidate = config_path or os.getenv("TRANSLATION_GRPO_CONFIG")
    if candidate:
        p = Path(candidate)
    else:
        p = Path(__file__).resolve().parent.parent / "configs" / "translation_grpo.yaml"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _get_shared_judge_client(judge_config: dict[str, Any] | None = None) -> AsyncOpenAIJudgeClient:
    sidecar = load_sidecar_config().get("judge", {})
    cfg_dict = {**(sidecar if isinstance(sidecar, dict) else {}), **dict(judge_config or {})}
    cfg = JudgeClientConfig(
        base_url=str(cfg_dict.get("base_url", os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1"))),
        api_key=str(cfg_dict.get("api_key", "")),
        api_key_env=str(cfg_dict.get("api_key_env", "JUDGE_API_KEY")),
        model=str(cfg_dict.get("model", os.getenv("JUDGE_MODEL", "Qwen3-32B"))),
        timeout_s=float(cfg_dict.get("timeout_s", 60.0)),
        max_retries=int(cfg_dict.get("max_retries", 4)),
        max_concurrency=int(cfg_dict.get("max_concurrency", 32)),
        temperature=float(cfg_dict.get("temperature", 0.0)),
        max_tokens=int(cfg_dict.get("max_tokens", 1024)),
        cache_path=cfg_dict.get("cache_path"),
        enable_cache=bool(cfg_dict.get("enable_cache", True)),
    )
    key = json.dumps(
        {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "cache_path": cfg.cache_path,
            "enable_cache": cfg.enable_cache,
        },
        sort_keys=True,
    )
    client = _SHARED_CLIENTS.get(key)
    if client is None:
        client = AsyncOpenAIJudgeClient(cfg)
        _SHARED_CLIENTS[key] = client
    return client


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """verl custom_reward_function entry point for asynchronous translation scoring.

    Calls the OpenAI-compatible online judge service (cached by SHA-256) and returns
    the 7-dimension reward plus per-dimension penalties, repetition/length flags,
    and optional anonymous reference-anchor score for downstream GRPO advantage shaping.
    """
    info = dict(extra_info or {})
    source = str(info.get("source", ""))
    direction = str(info.get("direction", data_source or "zh2en"))
    reference = str(info.get("reference") or ground_truth or "")
    source_units = info.get("source_len")
    response_token_len = info.get("response_len")

    sidecar_reward = load_sidecar_config().get("reward", {})
    reward_cfg_dict = {
        **(sidecar_reward if isinstance(sidecar_reward, dict) else {}),
        **dict(kwargs.get("reward_config") or {}),
    }
    reward_cfg = RewardWeightsConfig(
        **{k: v for k, v in reward_cfg_dict.items() if hasattr(RewardWeightsConfig, k)}
    )
    client: AsyncOpenAIJudgeClient = kwargs.get("judge_client") or _get_shared_judge_client(kwargs.get("judge_config"))
    score_anchor = bool(kwargs.get("score_anchor", True))

    if reward_cfg.reward_mode == "total_score":
        direct = await score_direct_total(client, source, solution_str, direction=direction)
        flags = detect_invalid_output(
            solution_str,
            source,
            source_units=source_units,
            response_token_len=response_token_len,
            config=reward_cfg,
        )
        # Scale 0-100 direct score into [-10, 0] range comparable with penalty rewards.
        score = (direct - 100.0) / 10.0
        result: dict[str, Any] = {
            "score": float(score),
            "direct_score": float(direct),
            "mqm_penalty": float(max(0.0, (100.0 - direct) / 10.0)),
            "accuracy_penalty": 0.0,
            "terminology_penalty": 0.0,
            "fluency_penalty": 0.0,
            "style_penalty": 0.0,
            "context_penalty": 0.0,
            "rep_penalty": 0.0,
            "len_penalty": 0.0,
            **flags,
        }
    else:
        try:
            mqm_res = await score_pointwise_mqm(client, source, solution_str, direction=direction)
        except JudgeOutputError as exc:
            # judge 多次输出不合格：该样本按 hard mask 处理，不参与组统计与 loss
            flags = detect_invalid_output(
                solution_str,
                source,
                source_units=source_units,
                response_token_len=response_token_len,
                config=reward_cfg,
            )
            return {
                "score": 0.0,
                **flags,
                "is_invalid": True,
                "judge_failed": True,
                "judge_error": str(exc)[:200],
            }
        result = score_candidate_from_pointwise(
            mqm_res,
            source,
            solution_str,
            source_units=source_units,
            response_token_len=response_token_len,
            config=reward_cfg,
        )

    # Optionally score the reference anchor anonymously using the identical judge protocol.
    # Because AsyncOpenAIJudgeClient deduplicates and caches identical prompts, all 8
    # candidates sharing the same uid reuse a single judge request for the reference anchor.
    if score_anchor and reference and reward_cfg.reward_mode != "total_score":
        try:
            anc_mqm = await score_pointwise_mqm(client, source, reference, direction=direction)
        except JudgeOutputError:
            return result
        anc_scored = score_candidate_from_pointwise(
            anc_mqm,
            source,
            reference,
            source_units=source_units,
            response_token_len=None,
            config=reward_cfg,
        )
        result["anchor_reward"] = float(anc_scored["score"])
        result["anchor_mqm_penalty"] = float(anc_scored["mqm_penalty"])
    else:
        result["anchor_reward"] = None
        result["anchor_mqm_penalty"] = None

    # Serialize nested error list as JSON string per heterogeneous parquet/TQ safety rules.
    if "mqm_errors" in result:
        result["mqm_errors_json"] = json.dumps(result.pop("mqm_errors"), ensure_ascii=False)
    return result
