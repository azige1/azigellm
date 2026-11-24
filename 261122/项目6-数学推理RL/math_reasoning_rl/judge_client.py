"""OpenAI-compatible per-subquestion LLM judge client with extract-then-verdict prompting."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from math_reasoning_rl.config import RewardRoutingConfig

# Note: Prompt templates use placeholder replacement instead of str.format()
# so JSON braces inside instructions never trigger KeyError/ValueError.
EXTRACT_THEN_JUDGE_PROMPT = """你是一位严谨的中小学数学阅卷老师。请针对题目中的单个小问，对学生作答进行严格判定。

【原题】
__QUESTION__

【待判小问】
- 小问编号：__SUBQ_ID__
- 标准答案：__STD_ANSWER__
- 关键步骤要求（若无则忽略）：__KEY_STEP__

【学生作答】
__STUDENT_RESPONSE__

【判定要求】
1. 先从学生作答中原文摘录该小问对应的最终答案与关键推导片段；若学生未作答该小问，摘录填 "未作答"。
2. 再比对标准答案与关键步骤要求：只有结论正确且推理/计算关键步骤正确（允许等价变形与不同合法解法，但不允许步骤缺失、跳步凑答案或关键公式/符号错误）时判定为 1，否则判定为 0。
3. 严格输出如下 JSON，不要输出额外文本：
{"extracted_answer": "摘录的该小问答案与关键步骤原文", "reason": "一句话比对说明", "verdict": 0或1}"""

DIRECT_SUBQ_JUDGE_PROMPT = """请判断学生作答在指定小问上是否正确。
原题：__QUESTION__
小问 __SUBQ_ID__ 标准答案：__STD_ANSWER__；关键步骤：__KEY_STEP__
学生作答：__STUDENT_RESPONSE__
输出 JSON：{"verdict": 0或1, "reason": "简要理由"}"""

WHOLE_PROBLEM_JUDGE_PROMPT = """请判断学生作答是否完整且正确地解答了题目中的所有小问。
原题：__QUESTION__
全部小问标准答案：__ALL_SUBQ_JSON__
学生作答：__STUDENT_RESPONSE__
输出 JSON：{"verdict": 0或1, "reason": "简要理由"}"""


@dataclass
class SubquestionVerdict:
    """Structured verdict for one subquestion."""

    subq_id: int | str
    extracted_answer: str
    verdict: int
    reason: str = ""
    raw_output: str = ""


def build_subquestion_judge_prompt(
    question: str,
    subq_item: dict[str, Any],
    student_response: str,
    strategy: str = "extract_then_judge",
) -> str:
    """Build judge prompt for one subquestion without using str.format on JSON braces."""
    subq_id = str(subq_item.get("id", 1))
    std_answer = str(subq_item.get("answer", ""))
    key_step = str(subq_item.get("key_step") or "无特定步骤要求")

    template = EXTRACT_THEN_JUDGE_PROMPT if strategy == "extract_then_judge" else DIRECT_SUBQ_JUDGE_PROMPT
    return (
        template.replace("__QUESTION__", question)
        .replace("__SUBQ_ID__", subq_id)
        .replace("__STD_ANSWER__", std_answer)
        .replace("__KEY_STEP__", key_step)
        .replace("__STUDENT_RESPONSE__", student_response)
    )


def build_whole_problem_judge_prompt(
    question: str,
    subq_items: list[dict[str, Any]],
    student_response: str,
) -> str:
    """Build baseline whole-problem prompt used in calibration comparisons."""
    return (
        WHOLE_PROBLEM_JUDGE_PROMPT.replace("__QUESTION__", question)
        .replace("__ALL_SUBQ_JSON__", json.dumps(subq_items, ensure_ascii=False))
        .replace("__STUDENT_RESPONSE__", student_response)
    )


def parse_judge_response(raw_text: str, subq_id: int | str = 1) -> SubquestionVerdict:
    """Parse structured JSON output from judge model with defensive regex fallback."""
    cleaned = raw_text.strip()
    # Strip optional thinking tags if judge emits reasoning preamble.
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned
    if not candidate.startswith("{"):
        brace_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)

    try:
        parsed = json.loads(candidate)
        extracted = str(parsed.get("extracted_answer", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        raw_verdict = parsed.get("verdict", 0)
        verdict = 1 if int(raw_verdict) == 1 else 0
        return SubquestionVerdict(
            subq_id=subq_id,
            extracted_answer=extracted,
            verdict=verdict,
            reason=reason,
            raw_output=raw_text,
        )
    except Exception:
        extracted_match = re.search(r'"extracted_answer"\s*:\s*"([^"]*)"', cleaned)
        verdict_match = re.search(r'"verdict"\s*:\s*([01])', cleaned)
        extracted = extracted_match.group(1).strip() if extracted_match else ""
        if verdict_match:
            verdict = int(verdict_match.group(1))
        else:
            # Conservative fallback: only mark 1 when explicitly stated, otherwise 0 to limit FP.
            verdict = 1 if re.search(r"\bverdict\s*[:=]\s*1\b", cleaned) else 0
        return SubquestionVerdict(
            subq_id=subq_id,
            extracted_answer=extracted,
            verdict=verdict,
            reason="regex_fallback",
            raw_output=raw_text,
        )


class OpenAICompatJudgeClient:
    """Async/sync client for OpenAI-compatible vLLM judge deployments."""

    def __init__(
        self,
        config: Optional[RewardRoutingConfig] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.config = config or RewardRoutingConfig.from_env()
        self._transport = transport

    def _endpoint_url(self) -> str:
        base = self.config.judge_base_url.rstrip("/")
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.judge_api_key}",
            "Content-Type": "application/json",
        }

    async def judge_prompt_async(self, prompt: str) -> str:
        """Send one chat-completion request to the judge server with retries."""
        payload = {
            "model": self.config.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
        }
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.config.judge_timeout_s, transport=self._transport) as client:
            for attempt in range(self.config.judge_max_retries):
                try:
                    resp = await client.post(self._endpoint_url(), json=payload, headers=self._headers())
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.config.judge_max_retries - 1:
                        await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Judge request failed after {self.config.judge_max_retries} retries: {last_exc}")

    async def judge_subquestions_async(
        self,
        question: str,
        subq_items: list[dict[str, Any]],
        student_response: str,
        strategy: str = "extract_then_judge",
    ) -> list[SubquestionVerdict]:
        """Evaluate every subquestion concurrently and return per-subquestion verdicts."""
        if not subq_items:
            return []

        async def _eval_one(item: dict[str, Any]) -> SubquestionVerdict:
            subq_id = item.get("id", 1)
            prompt = build_subquestion_judge_prompt(
                question=question,
                subq_item=item,
                student_response=student_response,
                strategy=strategy,
            )
            try:
                raw = await self.judge_prompt_async(prompt)
                return parse_judge_response(raw, subq_id=subq_id)
            except Exception as exc:
                return SubquestionVerdict(
                    subq_id=subq_id,
                    extracted_answer="",
                    verdict=0,
                    reason=f"judge_error:{exc}",
                )

        return list(await asyncio.gather(*(_eval_one(item) for item in subq_items)))

    def judge_subquestions_sync(
        self,
        question: str,
        subq_items: list[dict[str, Any]],
        student_response: str,
        strategy: str = "extract_then_judge",
    ) -> list[SubquestionVerdict]:
        """Synchronous wrapper for offline evaluation scripts and tests."""
        return asyncio.run(
            self.judge_subquestions_async(
                question=question,
                subq_items=subq_items,
                student_response=student_response,
                strategy=strategy,
            )
        )
