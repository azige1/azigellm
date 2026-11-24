import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class VerifierResult:
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("verifier 分数必须位于 [0,1]")


Verifier = Callable[[dict[str, Any], Any], Awaitable[VerifierResult]]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", str(text), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("响应中没有 JSON 对象")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("响应必须是 JSON 对象")
    return value


def task_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = kwargs.get("extra_info", {})
    raw = extra.get("task_payload_json", "{}") if isinstance(extra, dict) else "{}"
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def task_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    info = kwargs.get("extra_info", {})
    if info.get("task_payload_json"):
        return task_payload(kwargs)
    return {key: value for key, value in info.items() if key != "prompt_id"}


def single_turn_record(reply_text: str, length: int) -> dict[str, Any]:
    return {
        "reply_text": reply_text,
        "action_type": "final",
        "command": None,
        "response_start": 0,
        "response_end": length,
        "observation_start": None,
        "observation_end": None,
        "parse_valid": True,
    }
