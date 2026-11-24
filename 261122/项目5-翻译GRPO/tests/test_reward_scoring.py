"""Unit tests for 5.2 7-dim reward, 5.4 repetition/length detection, and async judge client."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from translation_grpo.judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig
from translation_grpo.reward_scoring import (
    RewardWeightsConfig,
    compute_ngram_repetition_rate,
    compute_score,
    compute_seven_dim_reward,
    detect_invalid_output,
)


def test_seven_dim_reward_numerical_check():
    # Spec 5.2 numerical verification:
    # Case 1: terminology minor (1) + fluency minor (1) -> -(1.5*1 + 1.0*1) = -2.5
    r1 = compute_seven_dim_reward(
        {"accuracy": 0.0, "terminology": 1.0, "fluency": 1.0, "style": 0.0, "context": 0.0},
        rep_flag=False,
        len_ratio=1.2,
        source_units=20,
    )
    assert r1["score"] == pytest.approx(-2.5)

    # Case 2: terminology major (5) -> -(1.5*5) = -7.5
    r2 = compute_seven_dim_reward(
        {"accuracy": 0.0, "terminology": 5.0, "fluency": 0.0, "style": 0.0, "context": 0.0},
        rep_flag=False,
        len_ratio=1.2,
        source_units=20,
    )
    assert r2["score"] == pytest.approx(-7.5)
    assert r2["score"] < r1["score"]


def test_length_penalty_exempt_under_10_units():
    # Source < 10 units -> no length penalty even if len_ratio > 1.5
    short = compute_seven_dim_reward(
        {"accuracy": 0.0, "terminology": 0.0, "fluency": 0.0, "style": 0.0, "context": 0.0},
        rep_flag=False,
        len_ratio=2.0,
        source_units=8,
    )
    assert short["len_penalty"] == pytest.approx(0.0)
    assert short["score"] == pytest.approx(0.0)

    # Source >= 10 units -> 5.0 * (2.0 - 1.5) = 2.5 penalty
    normal = compute_seven_dim_reward(
        {"accuracy": 0.0, "terminology": 0.0, "fluency": 0.0, "style": 0.0, "context": 0.0},
        rep_flag=False,
        len_ratio=2.0,
        source_units=12,
    )
    assert normal["len_penalty"] == pytest.approx(2.5)
    assert normal["score"] == pytest.approx(-2.5)


def test_repetition_detection_english_and_chinese():
    repeated_en = " ".join(["the contract is valid"] * 6)
    rate_en = compute_ngram_repetition_rate(repeated_en, n=4)
    assert rate_en > 0.3

    repeated_zh = "本合同自双方签字之日起生效" * 5
    rate_zh = compute_ngram_repetition_rate(repeated_zh, n=4)
    assert rate_zh > 0.3

    clean_zh = "本合同自双方签字盖章之日起生效，有效期三年，任何一方不得单方面解除。"
    assert compute_ngram_repetition_rate(clean_zh, n=4) == pytest.approx(0.0)


def test_invalid_output_triggers_on_rep_overlength_and_token_limit():
    cfg = RewardWeightsConfig()
    flags_rep = detect_invalid_output(" ".join(["repeat phrase here"] * 8), "source sentence with ten words total", source_units=10, config=cfg)
    assert flags_rep["rep_flag"] is True
    assert flags_rep["is_invalid"] is True

    long_cand = " ".join(["word"] * 30)
    flags_len = detect_invalid_output(long_cand, "one two three four five six seven eight nine ten", source_units=10, config=cfg)
    assert flags_len["len_ratio"] > 2.5
    assert flags_len["is_invalid"] is True

    flags_tok = detect_invalid_output("clean short text", "clean short text", source_units=3, response_token_len=1024, config=cfg)
    assert flags_tok["hit_max_tokens"] is True
    assert flags_tok["is_invalid"] is True


def test_async_judge_client_retry_and_cache(tmp_path):
    attempts = {"count": 0}

    async def _flaky_transport(payload):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise asyncio.TimeoutError("transient timeout")
        return json.dumps({"errors": [{"span": "term", "dim": "terminology", "severity": "minor"}]})

    cache_file = tmp_path / "judge_cache.jsonl"
    client = AsyncOpenAIJudgeClient(
        JudgeClientConfig(
            max_retries=3,
            retry_base_delay=0.001,
            cache_path=str(cache_file),
            enable_cache=True,
        ),
        transport=_flaky_transport,
    )

    extra = {
        "sample_id": "zh2en-001",
        "direction": "zh2en",
        "source": "本合同自双方签字盖章之日起生效，有效期三年，任何一方不得单方面解除。",
        "reference": "This contract takes effect upon signature and seal by both parties.",
        "source_len": 25,
        "response_len": 20,
    }
    res1 = asyncio.run(
        compute_score(
            data_source="translation_zh2en",
            solution_str="This contract takes effect upon signature.",
            ground_truth=extra["reference"],
            extra_info=extra,
            judge_client=client,
        )
    )
    # 1st candidate call failed once then succeeded (2), plus anchor call (1) = 3 transport calls.
    calls_after_first = attempts["count"]
    assert calls_after_first == 3
    assert res1["terminology_penalty"] == pytest.approx(1.0)
    assert res1["score"] == pytest.approx(-1.5)
    assert res1["anchor_reward"] == pytest.approx(-1.5)

    # Second call with identical inputs hits memory cache -> 0 new transport calls.
    res2 = asyncio.run(
        compute_score(
            data_source="translation_zh2en",
            solution_str="This contract takes effect upon signature.",
            ground_truth=extra["reference"],
            extra_info=extra,
            judge_client=client,
        )
    )
    assert attempts["count"] == calls_after_first
    assert res2["score"] == pytest.approx(res1["score"])
    assert cache_file.exists()
