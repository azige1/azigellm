"""压缩恢复样本：在累积阈值后的轮边界切分，后段历史由九段有损摘要重建。

切点之后至少还要 4 轮；恢复引导要求先确认再修改；摘要与引导不得出现
compaction、summary 等自指词；B 段重新做配对校验。
"""

import json
import re
from typing import Any, Optional

from trajectory_tools.pair_validation import validate_pairs
from trajectory_tools.token_estimate import count_messages_tokens, count_text_tokens
from trajectory_tools.prompts.nine_section_summary import nine_section_prompt
from agent_rollout.compact_history import CONTINUATION_PREFIX, find_last_read

BANNED_WORDS = re.compile(
    r"compaction|compact|summar(?:y|ize)|context limit|context window", re.I
)
RECOVERY_GUIDE = (
    "Continue the work directly. Before modifying any file, confirm the current "
    "content of the files you plan to change and run the test suite to check the "
    "current state; re-read any file whose content you are unsure about."
)
MIN_MESSAGES_AFTER_CUT = 4


def find_cut_point(
    messages: list[dict[str, Any]], cut_threshold: int = 50_000
) -> Optional[int]:
    """找切分点：返回切点处 assistant 消息的下标；不满足条件返回 None。

    切点要求：累计 token 超过阈值之后第一个轮边界（assistant 消息开始处，
    之前的调用都已返回），其工具调用含 Read 或测试命令，且切点后至少还有
    MIN_MESSAGES_AFTER_CUT 条消息。
    """
    from trajectory_tools.behavior_metrics import is_verification_command

    cumulative = 0
    crossed = -1
    for idx, message in enumerate(messages):
        if crossed >= 0 and idx >= crossed:
            break
        cumulative += count_text_tokens(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            cumulative += count_text_tokens(call.get("function", {}).get("arguments", ""))
        if cumulative > cut_threshold and crossed < 0:
            crossed = idx
    if crossed < 0:
        return None
    for idx in range(crossed, len(messages)):
        message = messages[idx]
        if message.get("role") != "assistant":
            continue
        # 切点之后（含切点）至少还要 MIN_MESSAGES_AFTER_CUT 个 assistant 轮次
        remaining_turns = sum(1 for m in messages[idx:] if m.get("role") == "assistant")
        if remaining_turns < MIN_MESSAGES_AFTER_CUT:
            return None
        has_read_or_test = False
        for call in message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name == "read_file":
                has_read_or_test = True
            elif name == "bash":
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if is_verification_command(args.get("command", "")):
                    has_read_or_test = True
        if has_read_or_test:
            return idx
    return None


def last_read_from_history(messages: list[dict[str, Any]]) -> tuple[Optional[str], str]:
    """从切点之前的历史里取最后一次 read_file 的路径与它的工具结果内容。"""
    call_to_path: dict[str, str] = {}
    last_path, last_content = None, ""
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") == "read_file":
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                call_to_path[call.get("id")] = args.get("path", "")
        if message.get("role") == "tool" and message.get("tool_call_id") in call_to_path:
            last_path = call_to_path[message["tool_call_id"]]
            last_content = message.get("content") or ""
    return last_path, last_content


def last_read_reminder(messages: list[dict[str, Any]]) -> str:
    """B 段摘要后附的提醒：最后一次读取的文件路径与截断到 2,000 字符的内容。"""
    path, content = last_read_from_history(messages)
    if not path:
        return ""
    return f"Last file read: {path}\n{content[:2000]}"


def summarize_prefix(
    messages: list[dict[str, Any]],
    client: Any,
    model: str,
    last_read_content: str = "",
) -> str:
    """对切点之前的历史生成九段摘要（附最后 Read 提醒）。"""
    if not last_read_content:
        last_read_content = last_read_from_history(messages)[1]
    history_text = "\n\n".join(
        f"[{m.get('role')}]\n{m.get('content') or ''}"
        + (
            "\n工具调用：" + "; ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})"
                for c in m.get("tool_calls") or []
            )
            if m.get("tool_calls")
            else ""
        )
        for m in messages
    )
    reminders = ""
    last_read_path = find_last_read(messages)[0]
    if last_read_path:
        reminders = (
            f"\n\n最后读取的文件：{last_read_path}\n内容（截断）：\n{last_read_content[:2000]}"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": nine_section_prompt()},
            {"role": "user", "content": history_text + reminders},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def pass_quality_gate(summary: str, recovery_guide: str = RECOVERY_GUIDE) -> bool:
    """摘要与恢复引导的自指词检查 + 引导的动作要求。"""
    if BANNED_WORDS.search(summary):
        return False
    if BANNED_WORDS.search(recovery_guide):
        return False
    lowered = recovery_guide.lower()
    return any(w in lowered for w in ("confirm", "re-read", "run the test"))


def assemble_part_b(
    traj: dict,
    cut_idx: int,
    summary: str,
    recovery_guide: str = RECOVERY_GUIDE,
) -> Optional[dict]:
    """组装 B 段：system + 任务 + 摘要 + 恢复引导 + 切点后的原始消息（保持 g）。"""
    messages = traj["messages"]
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    task_msg = next((m for m in messages if m.get("role") == "user"), None)
    if system_msg is None or task_msg is None:
        return None
    tail = [dict(m) for m in messages[cut_idx:]]
    summary_message = CONTINUATION_PREFIX + "\n\n" + summary
    reminder = last_read_reminder(messages[:cut_idx])
    if reminder:
        summary_message += "\n\n" + reminder
    part_b = {
        "id": traj.get("id", "") + "_B",
        "instance_id": traj.get("instance_id"),
        "messages": [
            dict(system_msg),
            dict(task_msg),
            {"role": "user", "content": summary_message},
            {"role": "user", "content": recovery_guide},
        ]
        + tail,
        "meta": dict(traj.get("meta", {})),
    }
    cut_token = count_messages_tokens(messages[:cut_idx])
    part_b["meta"]["segment"] = {"part": "B", "of": 2, "cut_token": cut_token}
    if validate_pairs(part_b["messages"]):
        return None
    return part_b


def compact_trajectory(
    traj: dict,
    client: Any,
    model: str,
    cut_threshold: int = 50_000,
    last_read_content: str = "",
) -> Optional[dict]:
    """对单条轨迹执行完整压缩恢复流程；不满足切分条件返回 None。"""
    messages = traj["messages"]
    cut_idx = find_cut_point(messages, cut_threshold)
    if cut_idx is None:
        return None
    summary = summarize_prefix(messages[:cut_idx], client, model, last_read_content)
    if not pass_quality_gate(summary):
        return None
    return assemble_part_b(traj, cut_idx, summary)


__all__ = []
