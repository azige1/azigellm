"""推理时压缩：与轨迹压缩恢复共用九段摘要模板，把长历史折叠成摘要 + 最近若干轮。"""

from pathlib import Path
from typing import Any

CONTINUATION_PREFIX = "This session is being continued from a previous conversation."
KEEP_RECENT_TURNS = 4
LAST_READ_CONTENT_CHARS = 2000


def recent_turns_start(messages: list[dict], keep_turns: int = KEEP_RECENT_TURNS) -> int:
    """最近 keep_turns 个 assistant 轮次的起始下标。

    保留段从 assistant 消息开始，保证每个工具结果都和它的调用一起保留，
    不会出现调用已被折叠、结果还留在上下文里的情况。
    """
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(assistant_indices) <= keep_turns:
        return assistant_indices[0] if assistant_indices else len(messages)
    return assistant_indices[-keep_turns]


def nine_section_prompt() -> str:
    """读取共享的九段摘要提示词模板。"""
    from trajectory_tools.prompts.nine_section_summary import nine_section_prompt as _prompt

    return _prompt()


def find_last_read(messages: list[dict]) -> tuple[str, str] | tuple[None, None]:
    """从消息历史里找最后一次 read_file 的文件路径与内容。"""
    last_path = None
    for message in reversed(messages):
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") == "read_file":
                try:
                    import json

                    args = json.loads(call["function"].get("arguments") or "{}")
                    last_path = args.get("path")
                except Exception:  # noqa: BLE001
                    continue
        if last_path:
            break
    return last_path, ""  # 内容在调用侧从工作区补充


def compact_messages(
    system_prompt: str,
    task_text: str,
    messages: list[dict],
    client: Any,
    model: str,
    last_read_content: str = "",
    keep_turns: int = KEEP_RECENT_TURNS,
    request_extra: dict | None = None,
) -> tuple[list[dict], str]:
    """对历史做九段摘要，返回 (新上下文, 摘要文本)。

    新上下文 = 系统提示 + 任务 + 摘要消息 + 最近 keep_turns 个完整轮次。
    messages 前两条是系统提示与任务，不进入摘要正文。
    """
    keep_from = recent_turns_start(messages, keep_turns)
    history = messages[2:keep_from]
    recent = messages[keep_from:]
    history_text = "\n\n".join(
        f"[{m.get('role', '?')}]\n{m.get('content') or ''}"
        + (
            "\n工具调用：" + "; ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})"
                for c in m.get("tool_calls") or []
            )
            if m.get("tool_calls")
            else ""
        )
        for m in history
    )
    reminders = ""
    last_read_path = find_last_read(history)[0]
    if last_read_path:
        reminders = (
            f"\n\n最后读取的文件：{last_read_path}\n"
            f"内容（截断）：\n{last_read_content[:LAST_READ_CONTENT_CHARS]}"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": nine_section_prompt()},
            {"role": "user", "content": history_text + reminders},
        ],
        temperature=0.3,
        max_tokens=2048,
        extra_body=request_extra or None,
    )
    summary = response.choices[0].message.content or ""
    new_context = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
        {"role": "user", "content": CONTINUATION_PREFIX + "\n\n" + summary},
    ] + [dict(m) for m in recent]
    return new_context, summary


__all__ = []
