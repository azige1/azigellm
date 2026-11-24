"""配对校验：tool_calls 与 tool 消息的一一对应、顺序与悬空检查。

轨迹消息为带 ``g``（全局索引）的 OpenAI 风格消息列表；校验失败返回
错误类型列表，空列表表示通过。
"""

from typing import Any

ERROR_MISSING_RESULT = "missing_result"
ERROR_ORPHAN_RESULT = "orphan_result"
ERROR_ORDER_VIOLATION = "order_violation"
ERROR_DANGLING_CALL = "dangling_call"


def validate_pairs(messages: list[dict[str, Any]]) -> list[str]:
    """三项校验：每个调用恰有一个结果；结果在调用之后、下一条 assistant 之前；末尾无悬空调用。"""
    errors: list[str] = []
    pending: dict[str, int] = {}  # call_id -> 所在 assistant 的消息下标
    answered: set[str] = set()
    last_assistant_idx = -1
    for idx, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id in pending or call_id in answered:
                    errors.append(ERROR_MISSING_RESULT)
                pending[call_id] = idx
            last_assistant_idx = idx
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                if call_id in answered:
                    errors.append(ERROR_MISSING_RESULT)  # 重复结果
                else:
                    errors.append(ERROR_ORPHAN_RESULT)
            else:
                owner = pending.pop(call_id)
                # 结果必须紧跟其调用：位于调用之后、下一条 assistant 之前
                if idx < owner or owner != last_assistant_idx:
                    errors.append(ERROR_ORDER_VIOLATION)
                answered.add(call_id)
    for call_id in pending:
        errors.append(ERROR_DANGLING_CALL)
    return errors


__all__ = []
