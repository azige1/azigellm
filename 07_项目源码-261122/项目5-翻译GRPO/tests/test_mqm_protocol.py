"""Unit tests for 5.1 hybrid Pointwise MQM + position-swapped Pairwise protocol."""

from __future__ import annotations

import asyncio
import json

import pytest

from translation_grpo.calibration import load_gold_pairs, run_calibration_suite
from translation_grpo.judge_client import (
    PAIRWISE_JUDGE_PROMPT_TEMPLATE,
    POINTWISE_MQM_PROMPT_TEMPLATE,
    AsyncOpenAIJudgeClient,
    JudgeClientConfig,
    extract_json_object,
    render_prompt_template,
)
from translation_grpo.mqm_protocol import (
    MQMErrorSpan,
    compare_pairwise_swapped,
    compute_mqm_penalty,
    hybrid_compare_pair,
    normalize_dimension,
    normalize_severity,
    score_group_with_anchor_hybrid,
    score_pointwise_mqm,
)


def test_prompt_templates_render_with_json_braces():
    rendered_pw = render_prompt_template(
        POINTWISE_MQM_PROMPT_TEMPLATE,
        {"__DIRECTION__": "zh2en", "__SOURCE__": "合同自签字之日起生效。", "__CANDIDATE__": "The contract {is} valid."},
    )
    assert '"errors"' in rendered_pw
    assert "__SOURCE__" not in rendered_pw

    rendered_pair = render_prompt_template(
        PAIRWISE_JUDGE_PROMPT_TEMPLATE,
        {
            "__DIRECTION__": "en2zh",
            "__SOURCE__": "Test {source}",
            "__CANDIDATE_A__": "A {text}",
            "__CANDIDATE_B__": "B {text}",
        },
    )
    assert '"winner"' in rendered_pair
    assert "__CANDIDATE_A__" not in rendered_pair


def test_extract_json_object_handles_fences_and_prose():
    raw = 'Here is my analysis:\n```json\n{"errors": [{"span": "x", "dim": "terminology", "severity": "major"}]}\n```'
    parsed = extract_json_object(raw)
    assert len(parsed["errors"]) == 1
    assert parsed["errors"][0]["dim"] == "terminology"


def test_mqm_penalty_weights_and_normalization():
    assert normalize_dimension("本地化") == "context"
    assert normalize_dimension("术语") == "terminology"
    assert normalize_severity("严重") == "critical"
    assert normalize_severity("主要") == "major"
    assert normalize_severity("轻微") == "minor"

    errors = [
        MQMErrorSpan(span="a", dim="terminology", severity="critical", weight=25.0),
        MQMErrorSpan(span="b", dim="accuracy", severity="major", weight=5.0),
        MQMErrorSpan(span="c", dim="fluency", severity="minor", weight=1.0),
    ]
    total, dims = compute_mqm_penalty(errors)
    assert total == pytest.approx(31.0)
    assert dims["terminology"] == pytest.approx(25.0)
    assert dims["accuracy"] == pytest.approx(5.0)
    assert dims["fluency"] == pytest.approx(1.0)
    assert dims["style"] == pytest.approx(0.0)
    assert dims["context"] == pytest.approx(0.0)


def test_pairwise_swapped_position_bias_becomes_tie():
    async def _mock_transport(payload):
        # Always favors whichever translation appears first ("Translation A").
        return json.dumps({"winner": "A", "reason": "first option"})

    client = AsyncOpenAIJudgeClient(
        JudgeClientConfig(enable_cache=False),
        transport=_mock_transport,
    )
    res = asyncio.run(compare_pairwise_swapped(client, "源句", "译文甲", "译文乙", "zh2en"))
    assert res.winner == "tie"
    assert res.consistent is False
    assert res.position_bias_triggered is True


def test_pairwise_swapped_consistent_winner():
    async def _mock_transport(payload):
        content = payload["messages"][-1]["content"]
        # Identify better candidate regardless of A/B slot.
        marker_a = content.split("Translation A:\n")[1].split("\n\nTranslation B:")[0].strip()
        if marker_a == "GOOD":
            return json.dumps({"winner": "A"})
        return json.dumps({"winner": "B"})

    client = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_mock_transport)
    res_ab = asyncio.run(compare_pairwise_swapped(client, "src", "GOOD", "BAD", "zh2en"))
    assert res_ab.winner == "A"
    assert res_ab.consistent is True
    assert res_ab.position_bias_triggered is False


def test_hybrid_threshold_and_group_shuffle_with_anchor():
    calls = []

    async def _mock_transport(payload):
        msg = payload["messages"][-1]["content"]
        calls.append(msg)
        if "Translation A:" in msg:
            slot_a = msg.split("Translation A:\n")[1].split("\n\nTranslation B:")[0].strip()
            return json.dumps({"winner": "A" if slot_a == "cand0" else "B"})
        if "cand1" in msg:
            return json.dumps({"errors": [{"span": "x", "dim": "fluency", "severity": "minor"}]})
        if "cand2" in msg:
            return json.dumps({"errors": [{"span": "y", "dim": "terminology", "severity": "major"}]})
        return json.dumps({"errors": []})

    client = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_mock_transport)

    # |0 - 5| = 5 >= 2 -> decided by pointwise without pairwise call
    hyb_far = asyncio.run(hybrid_compare_pair(client, "src", "cand0", "cand2", "zh2en", diff_threshold=2.0))
    assert hyb_far.decided_by == "pointwise"
    assert hyb_far.winner == "A"

    # |0 - 1| = 1 < 2 -> escalated to position-swapped pairwise
    hyb_close = asyncio.run(hybrid_compare_pair(client, "src", "cand0", "cand1", "zh2en", diff_threshold=2.0))
    assert hyb_close.decided_by == "pairwise"
    assert hyb_close.winner == "A"

    # 8 candidates + 1 anchor shuffled anonymously
    cands = ["cand0", "cand1", "cand2", "cand3", "cand4", "cand5", "cand6", "cand7"]
    grp = asyncio.run(
        score_group_with_anchor_hybrid(
            client,
            "src",
            cands,
            direction="zh2en",
            reference_anchor="anchor_ref",
            enable_pairwise_tiebreak=True,
            diff_threshold=2.0,
            seed=123,
        )
    )
    assert len(grp.candidate_results) == 8
    assert grp.anchor_result is not None
    assert grp.anchor_result.total_penalty == pytest.approx(0.0)
    assert grp.candidate_results[2].total_penalty == pytest.approx(5.0)
    assert grp.pairwise_calls > 0


def test_calibration_suite_on_sample_gold_file(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps({"source": "s1", "candidate_a": "GOOD", "candidate_b": "BAD", "gold_winner": "A"}) + "\n",
        encoding="utf-8",
    )
    gold = load_gold_pairs(gold_path)

    async def _mock_transport(payload):
        msg = payload["messages"][-1]["content"]
        if "score from 0 to 100" in msg:
            return json.dumps({"score": 90 if "GOOD" in msg else 60})
        if "Translation A:" in msg:
            slot_a = msg.split("Translation A:\n")[1].split("\n\nTranslation B:")[0].strip()
            return json.dumps({"winner": "A" if slot_a == "GOOD" else "B"})
        if "BAD" in msg:
            return json.dumps({"errors": [{"span": "bad", "dim": "terminology", "severity": "major"}]})
        return json.dumps({"errors": []})

    client = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_mock_transport)
    suite = asyncio.run(run_calibration_suite(client, gold))
    assert len(suite) == 4
    for m in suite:
        assert m.agreement_excl_ties == pytest.approx(1.0)


def test_invalid_mqm_output_is_retried_then_rejected():
    from translation_grpo.judge_client import JudgeOutputError
    from translation_grpo.mqm_protocol import score_pointwise_mqm

    calls = {"n": 0}

    async def _mock_transport(payload):
        calls["n"] += 1
        return "{}"  # 缺少 errors 字段，不能被当作「没有错误」

    client = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=True), transport=_mock_transport)
    with pytest.raises(JudgeOutputError):
        asyncio.run(score_pointwise_mqm(client, "源句", "译文", "zh2en"))
    assert calls["n"] == 3


def test_empty_error_list_is_valid_zero_penalty():
    from translation_grpo.mqm_protocol import score_pointwise_mqm

    async def _mock_transport(payload):
        return json.dumps({"errors": []})

    client = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_mock_transport)
    res = asyncio.run(score_pointwise_mqm(client, "源句", "译文", "zh2en"))
    assert res.total_penalty == 0.0
