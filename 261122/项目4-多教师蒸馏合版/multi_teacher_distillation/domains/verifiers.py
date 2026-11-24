import json
import math
import os
import re
import unicodedata
from typing import Any

import httpx

from .. import EVAL_JUDGE_MODEL, TRAIN_JUDGE_MODEL
from .common import VerifierResult, parse_json_object


def normalized_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"[\s\u3000]+", " ", text)
    return text.strip(" \t\r\n.,;:!?，。；：！？")


def _extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    for line in reversed(text.splitlines()):
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return text.strip()


def _numeric_or_symbolic_match(actual: str, expected: str, tol: float = 1e-6) -> bool:
    if normalized_answer(actual) == normalized_answer(expected):
        return True
    try:
        if math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
            return True
    except (TypeError, ValueError):
        pass
    try:
        import sympy as sp

        diff = sp.simplify(sp.sympify(actual) - sp.sympify(expected))
        return diff == 0
    except Exception:
        return False


async def verify_math_science(task: dict[str, Any], response: Any, **kwargs) -> VerifierResult:
    del kwargs
    expected = task.get("answer", task.get("expected_answer", ""))
    extracted = _extract_final_answer(str(response))
    accepted = task.get("answers", [expected])
    matched = any(
        _numeric_or_symbolic_match(extracted, str(cand), float(task.get("tolerance", 1e-6)))
        for cand in accepted
        if cand is not None
    )
    return VerifierResult(float(matched), matched, {"extracted": extracted, "expected": expected})


def _check_ifeval_constraint(text: str, constraint: dict[str, Any]) -> bool:
    ctype = str(constraint.get("type", ""))
    if ctype == "max_words":
        return len(text.split()) <= int(constraint["limit"])
    if ctype == "min_words":
        return len(text.split()) >= int(constraint["limit"])
    if ctype == "include_keyword":
        return str(constraint["keyword"]) in text
    if ctype == "exclude_keyword":
        return str(constraint["keyword"]) not in text
    if ctype == "json_format":
        try:
            parse_json_object(text)
            return True
        except ValueError:
            return False
    if ctype == "min_paragraphs":
        paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        return len(paras) >= int(constraint["limit"])
    if ctype == "regex_match":
        return bool(re.search(str(constraint["pattern"]), text))
    return False


async def verify_instruction_following(task: dict[str, Any], response: Any, **kwargs) -> VerifierResult:
    del kwargs
    text = str(response)
    constraints = list(task.get("constraints", []))
    if not constraints:
        expected = task.get("answer")
        passed = True if expected is None else normalized_answer(text) == normalized_answer(expected)
        return VerifierResult(float(passed), passed, {"constraints_checked": 0})
    results = [_check_ifeval_constraint(text, item) for item in constraints]
    strict_pass = all(results)
    score = sum(results) / len(results)
    return VerifierResult(score if task.get("soft_score", False) else float(strict_pass), strict_pass, {
        "constraints_passed": sum(results),
        "constraints_total": len(results),
    })


async def verify_tool_calling(task: dict[str, Any], response: Any, **kwargs) -> VerifierResult:
    del kwargs
    try:
        payload = response if isinstance(response, dict) else parse_json_object(str(response))
        expected_calls = task.get("expected_calls", [])
        actual_calls = payload.get("tool_calls", [payload] if "name" in payload else [])
        if not expected_calls:
            expected_name = task.get("tool_name")
            passed = expected_name is not None and payload.get("name") == expected_name
            return VerifierResult(float(passed), passed, {"actual_calls": actual_calls})
        if len(actual_calls) != len(expected_calls):
            return VerifierResult(0.0, False, {"reason": "call_count_mismatch"})
        matched = 0
        for actual, expected in zip(actual_calls, expected_calls, strict=True):
            same_name = actual.get("name") == expected.get("name")
            expected_args = dict(expected.get("arguments", {}))
            actual_args = dict(actual.get("arguments", {}))
            same_args = all(actual_args.get(k) == v for k, v in expected_args.items())
            if same_name and same_args:
                matched += 1
        passed = matched == len(expected_calls)
        return VerifierResult(float(passed), passed, {"matched_calls": matched, "total_calls": len(expected_calls)})
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return VerifierResult(0.0, False, {"error": str(exc)})


async def verify_execution_pass_rate(task: dict[str, Any], response: Any, **kwargs) -> VerifierResult:
    """代码/仓库/issue/测试生成等执行类领域的通过率判分接口（支持预注入 test_runner 或已计算结果）。"""
    if isinstance(response, dict) and "passed_tests" in response and "total_tests" in response:
        total = max(int(response["total_tests"]), 1)
        passed_cnt = max(0, min(int(response["passed_tests"]), total))
        tampered = bool(response.get("tamper_flag", False))
        score = 0.0 if tampered else passed_cnt / float(total)
        return VerifierResult(score, (not tampered) and score >= float(task.get("pass_threshold", 1.0)), {
            "passed_tests": passed_cnt,
            "total_tests": total,
            "tamper_flag": tampered,
        })
    runner = kwargs.get("test_runner")
    if runner is not None:
        outcome = await runner(task, response)
        total = max(int(outcome["total_tests"]), 1)
        passed_cnt = int(outcome["passed_tests"])
        tampered = bool(outcome.get("tamper_flag", False))
        score = 0.0 if tampered else passed_cnt / float(total)
        return VerifierResult(score, (not tampered) and score >= float(task.get("pass_threshold", 0.5)), outcome)
    expected_substrings = task.get("expected_substrings", [])
    if expected_substrings:
        text = str(response)
        hits = sum(1 for item in expected_substrings if str(item) in text)
        score = hits / len(expected_substrings)
        return VerifierResult(score, hits == len(expected_substrings), {"matched": hits, "total": len(expected_substrings)})
    raise RuntimeError("执行类领域判分需要 test_runner 或执行统计字段")


class PairwiseJudge:
    """通用对话成对比较 judge：交换位置各判一次，两次一致才计分。"""

    def __init__(
        self,
        base_url: str,
        model: str = TRAIN_JUDGE_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY")
        self.timeout = timeout

    async def _judge_once(self, prompt_text: str, candidate_a: str, candidate_b: str) -> str:
        payload = json.dumps({
            "instruction": "比较回答 A 与回答 B 的质量，只返回 JSON: {\"winner\": \"A\" | \"B\" | \"tie\"}",
            "prompt": prompt_text,
            "response_a": candidate_a,
            "response_b": candidate_b,
        }, ensure_ascii=False)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": [{"role": "user", "content": payload}], "temperature": 0},
            )
            resp.raise_for_status()
        parsed = parse_json_object(resp.json()["choices"][0]["message"]["content"])
        winner = str(parsed["winner"]).strip().upper()
        if winner not in {"A", "B", "TIE"}:
            raise ValueError(f"非法 pairwise winner: {winner}")
        return winner

    async def score(self, task: dict[str, Any], response: Any) -> VerifierResult:
        prompt_text = str(task.get("prompt_text", task.get("prompt", "")))
        reference_answer = str(task.get("reference_answer", task.get("answer", "")))
        student_answer = str(response)
        first = await self._judge_once(prompt_text, student_answer, reference_answer)
        second = await self._judge_once(prompt_text, reference_answer, student_answer)
        consistent_win = first == "A" and second == "B"
        consistent_tie = first == "TIE" and second == "TIE"
        consistent_loss = first == "B" and second == "A"
        if consistent_win:
            score, passed = 1.0, True
        elif consistent_tie:
            score, passed = 0.5, False
        else:
            score, passed = 0.0, False
        return VerifierResult(score, passed, {
            "first_order": first,
            "swapped_order": second,
            "consistent": consistent_win or consistent_tie or consistent_loss,
        })


async def verify_general_chat(task: dict[str, Any], response: Any, config: Any = None, **kwargs) -> VerifierResult:
    del kwargs
    settings = getattr(config, "judge", None) if config is not None else None
    base_url = str(getattr(settings, "base_url", os.environ.get("JUDGE_BASE_URL", "")))
    if not base_url:
        if "reference_win" in task:
            win = bool(task["reference_win"])
            return VerifierResult(float(win), win, {"offline": True})
        raise RuntimeError("通用对话判分需要 JUDGE_BASE_URL")
    judge = PairwiseJudge(
        base_url=base_url,
        model=str(getattr(settings, "model", TRAIN_JUDGE_MODEL)),
        timeout=float(getattr(settings, "timeout", 120.0)),
    )
    return await judge.score(task, response)


async def verify_self_anchor(task: dict[str, Any], response: Any, **kwargs) -> VerifierResult:
    del task, response, kwargs
    return VerifierResult(1.0, True, {"self_anchor": True})


def verifier_for(domain: str, config: Any = None):
    verifiers = {
        "code_gen": verify_execution_pass_rate,
        "repo_impl": verify_execution_pass_rate,
        "issue_fix": verify_execution_pass_rate,
        "test_gen": verify_execution_pass_rate,
        "math_science": verify_math_science,
        "instruction_following": verify_instruction_following,
        "tool_calling": verify_tool_calling,
        "general_chat": verify_general_chat,
        "self_anchor": verify_self_anchor,
    }
    if domain not in verifiers:
        raise ValueError(f"未知领域 verifier: {domain}")
    verifier = verifiers[domain]

    async def bound(task: dict[str, Any], response: Any, **kwargs: Any) -> VerifierResult:
        return await verifier(task, response, config=config, **kwargs)

    return bound


__all__ = [
    "EVAL_JUDGE_MODEL",
    "PairwiseJudge",
    "TRAIN_JUDGE_MODEL",
    "normalized_answer",
    "verifier_for",
    "verify_execution_pass_rate",
    "verify_general_chat",
    "verify_instruction_following",
    "verify_math_science",
    "verify_self_anchor",
    "verify_tool_calling",
]
