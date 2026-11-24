"""Gated rule-or-judge reward scoring compatible with verl 0.9.1 RewardManager."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from math_reasoning_rl.config import RewardRoutingConfig
from math_reasoning_rl.judge_client import OpenAICompatJudgeClient, SubquestionVerdict
from math_reasoning_rl.symbolic_verify import verify_rule_answer

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s\w]")


def split_thinking_and_answer(response_str: str) -> tuple[Optional[str], str, bool]:
    """Split Qwen3 thinking-mode response into (thinking, answer, tags_balanced).

    Returns ``tags_balanced=False`` when ``<think>``/``</think>`` tags are malformed.
    """
    text = response_str or ""
    open_count = text.count("<think>")
    close_count = text.count("</think>")

    if open_count == 0 and close_count == 0:
        return None, text.strip(), True

    if open_count > 1 or close_count != 1:
        return None, "", False

    if open_count == 1:
        start_idx = text.find("<think>") + len("<think>")
        end_idx = text.find("</think>")
        if end_idx < start_idx:
            return None, "", False
        thinking = text[start_idx:end_idx].strip()
        answer = text[end_idx + len("</think>") :].strip()
        return thinking, answer, True

    # Some chat templates inject `<think>` inside the prompt so only `</think>` appears in response.
    end_idx = text.find("</think>")
    thinking = text[:end_idx].strip()
    answer = text[end_idx + len("</think>") :].strip()
    return thinking, answer, True


def compute_ngram_repetition_rate(text: str, n: int = 4) -> float:
    """Compute n-gram repetition rate ``1 - |unique_ngrams| / |all_ngrams|``."""
    if not text or not text.strip():
        return 0.0
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def check_response_gate(
    response_str: str,
    config: Optional[RewardRoutingConfig] = None,
) -> tuple[bool, str, Optional[str], str, float, float]:
    """Check format compliance and 4-gram repetition gates on thinking & answer.

    Returns:
        (passed, reason, thinking, answer, rep_rate_think, rep_rate_answer)
    """
    cfg = config or RewardRoutingConfig.from_env()
    thinking, answer, tags_balanced = split_thinking_and_answer(response_str)

    if not tags_balanced:
        return False, "unbalanced_think_tags", thinking, answer, 1.0, 1.0
    if cfg.require_think_tags and (thinking is None or not thinking.strip()):
        return False, "missing_thinking_segment", thinking, answer, 0.0, 0.0
    if not answer or not answer.strip():
        return False, "empty_answer_segment", thinking, answer, 0.0, 0.0

    rep_think = compute_ngram_repetition_rate(thinking or "", n=cfg.repetition_ngram_n)
    rep_answer = compute_ngram_repetition_rate(answer, n=cfg.repetition_ngram_n)
    if rep_think > cfg.repetition_threshold:
        return False, "thinking_repetition_exceeded", thinking, answer, rep_think, rep_answer
    if rep_answer > cfg.repetition_threshold:
        return False, "answer_repetition_exceeded", thinking, answer, rep_think, rep_answer

    return True, "ok", thinking, answer, rep_think, rep_answer


def _parse_subq_metadata(ground_truth: Any, extra_info: Optional[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    """Resolve routing mode, question text, and subquestion list from verl inputs."""
    info = extra_info or {}
    question = str(info.get("question", ""))
    subq_raw = info.get("subq") if "subq" in info else info.get("subq_json")

    if isinstance(subq_raw, str):
        try:
            subq_list = json.loads(subq_raw)
        except json.JSONDecodeError:
            subq_list = []
    elif isinstance(subq_raw, list):
        subq_list = subq_raw
    else:
        subq_list = []

    if not subq_list and isinstance(ground_truth, str) and ground_truth.strip().startswith("{"):
        try:
            gt_obj = json.loads(ground_truth)
            if isinstance(gt_obj, dict) and "subq" in gt_obj:
                subq_list = gt_obj["subq"]
                question = question or str(gt_obj.get("question", ""))
        except json.JSONDecodeError:
            pass

    if not subq_list:
        subq_list = [{"id": 1, "answer": str(ground_truth)}]

    explicit_route = info.get("route")
    if explicit_route in ("rule", "judge"):
        route = str(explicit_route)
    else:
        problem_type = str(info.get("type", ""))
        if len(subq_list) > 1 or problem_type == "解答":
            route = "judge"
        else:
            route = "rule"

    return route, question, subq_list


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    judge_client: Optional[OpenAICompatJudgeClient] = None,
    config: Optional[RewardRoutingConfig] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl 0.9.1 custom_reward_function entry point (async-native for RewardLoopWorker).

    Flow (6.1):
    1. Gate: format compliance + <= 0.3 4-gram repetition on both thinking and answer.
       Any gate failure returns reward -1.0 without entering correctness scoring.
    2. Route: single-expression items go to SymPy symbolic verification; multi-subquestion
       or long-form items go to per-subquestion extract-then-verdict judge.
    3. Verdict: reward is +1.0 iff all subquestions pass, otherwise -1.0.
    """
    cfg = config or _config_from_kwargs(kwargs) or RewardRoutingConfig.from_env()
    passed_gate, gate_reason, _thinking, answer, rep_think, rep_answer = check_response_gate(solution_str, cfg)
    route, question, subq_list = _parse_subq_metadata(ground_truth, extra_info)

    if not passed_gate:
        return {
            "score": -1.0,
            "acc": 0.0,
            "route": route,
            "gate_passed": 0.0,
            "gate_reason": gate_reason,
            "rep_4gram_think": float(rep_think),
            "rep_4gram_ans": float(rep_answer),
            "sub_verdict": json.dumps([]),
        }

    if route == "rule":
        std_ans = str(subq_list[0].get("answer", ground_truth))
        is_correct, extracted = verify_rule_answer(answer, std_ans)
        verdicts = [1 if is_correct else 0]
        reward = 1.0 if is_correct else -1.0
        return {
            "score": reward,
            "acc": 1.0 if is_correct else 0.0,
            "route": "rule",
            "gate_passed": 1.0,
            "gate_reason": "ok",
            "rep_4gram_think": float(rep_think),
            "rep_4gram_ans": float(rep_answer),
            "sub_verdict": json.dumps(verdicts),
            "extracted_pred": extracted or "",
        }

    client = judge_client or OpenAICompatJudgeClient(cfg)
    sub_verdicts: list[SubquestionVerdict] = await client.judge_subquestions_async(
        question=question,
        subq_items=subq_list,
        student_response=answer,
        strategy="extract_then_judge",
    )
    verdict_bits = [v.verdict for v in sub_verdicts]
    all_correct = len(verdict_bits) == len(subq_list) and len(verdict_bits) > 0 and all(v == 1 for v in verdict_bits)
    reward = 1.0 if all_correct else -1.0

    return {
        "score": reward,
        "acc": 1.0 if all_correct else 0.0,
        "route": "judge",
        "gate_passed": 1.0,
        "gate_reason": "ok",
        "rep_4gram_think": float(rep_think),
        "rep_4gram_ans": float(rep_answer),
        "sub_verdict": json.dumps(verdict_bits),
        "judge_extractions": json.dumps([v.extracted_answer for v in sub_verdicts], ensure_ascii=False),
    }


def _config_from_kwargs(kwargs: dict[str, Any]) -> Optional[RewardRoutingConfig]:
    """Construct RewardRoutingConfig when overridden via custom_reward_function.reward_kwargs."""
    keys = {
        "repetition_ngram_n",
        "repetition_threshold",
        "require_think_tags",
        "judge_base_url",
        "judge_api_key",
        "judge_model",
        "judge_timeout_s",
        "judge_max_retries",
    }
    filtered = {k: kwargs[k] for k in keys if k in kwargs and kwargs[k] is not None}
    if not filtered:
        return None
    base = RewardRoutingConfig.from_env()
    for k, v in filtered.items():
        setattr(base, k, v)
    return base


def compute_score_sync(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    judge_client: Optional[OpenAICompatJudgeClient] = None,
    config: Optional[RewardRoutingConfig] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous helper around ``compute_score`` for scripts and unit tests."""
    return asyncio.run(
        compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            judge_client=judge_client,
            config=config,
            **kwargs,
        )
    )
