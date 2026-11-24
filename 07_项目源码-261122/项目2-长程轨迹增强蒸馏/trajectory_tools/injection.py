"""过程监督注入：在成功轨迹的关键决策点前置英文状态回顾。

触发点九主两备；预算按调和平均式分配，完成宣告必选且不占预算；
质量门纯规则；写回只改 assistant 文本内容，不动 tool_calls。
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from trajectory_tools.token_estimate import count_text_tokens
from trajectory_tools.behavior_metrics import (
    WRITE_EDIT_TOOLS,
    is_verification_command,
    iter_tool_calls,
)

MAX_INJECTIONS = 10
MIN_REVIEW_WORDS = 40
MAX_REVIEW_WORDS = 120

COMPLETION_PATTERN = re.compile(r"TASK COMPLETE|all tests pass|task is complete", re.I)
STATE_PHRASES = (
    "completed",
    "remaining",
    "not yet verified",
    "still need",
    "verified by",
    "in progress",
    "done",
)
VERIFICATION_WORDS = ("test", "pytest", "ran", "passed")
IMPERATIVE_PATTERN = re.compile(
    r"^\s*(Run|Open|Edit|Create|Write|Add|Fix|Read|Check|Update|Move|Delete|Install|Navigate)\b",
    re.M,
)

# 主触发点优先级（贪心选取时从高到低）
PRIORITY = [
    "error_to_fix",
    "first_test_run",
    "before_last_test_run",
    "blind_segment_end",
    "blind_segment_start",
    "first_edit_after_plan",
    "phase_boundary",
    "plan_update",
    "completion_claim",
]
TEST_RELATED_TRIGGERS = {"error_to_fix", "first_test_run", "before_last_test_run"}


@dataclass
class Trigger:
    """一个注入触发点。"""

    g: int
    type: str


def _parse_args(call: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(call.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}


def _blind_runs(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """返回 (首条写消息下标, 末条写消息下标) 的盲改段列表（长度 >=3）。"""
    runs: list[tuple[int, int]] = []
    start = None
    count = 0
    last = None
    for idx, call in iter_tool_calls(messages):
        name = call.get("function", {}).get("name")
        if name in WRITE_EDIT_TOOLS:
            if start is None:
                start = idx
            count += 1
            last = idx
        elif name == "bash" and is_verification_command(_parse_args(call).get("command", "")):
            if start is not None and count >= 3:
                runs.append((start, last))
            start, count, last = None, 0, None
    if start is not None and count >= 3:
        runs.append((start, last))
    return runs


def find_trigger_points(messages: list[dict[str, Any]]) -> list[Trigger]:
    """沿 g 扫描识别九类主触发点；主触发点少于 3 个时启用两类备选。"""
    triggers: list[Trigger] = []
    seen: set[tuple[int, str]] = set()

    def add(g: int, ttype: str) -> None:
        if (g, ttype) not in seen and g is not None:
            seen.add((g, ttype))
            triggers.append(Trigger(g=g, type=ttype))

    # 工具调用序列的辅助视图
    calls = [(idx, call) for idx, call in iter_tool_calls(messages)]
    verification_assistants = [
        idx for idx, call in calls if call.get("function", {}).get("name") == "bash"
        and is_verification_command(_parse_args(call).get("command", ""))
    ]
    todo_history: list[tuple[int, int]] = []  # (assistant 下标, 项数)

    # 盲改段
    blind = _blind_runs(messages)
    for start, end in blind:
        add(messages[start].get("g"), "blind_segment_start")
        add(messages[end].get("g"), "blind_segment_end")

    # 计划类
    first_edit_after_plan_done = False
    for idx, call in calls:
        name = call.get("function", {}).get("name")
        if name == "todo_write":
            add(messages[idx].get("g"), "plan_update")
            items = _parse_args(call).get("items") or []
            if todo_history and len(items) < todo_history[-1][1]:
                add(messages[idx].get("g"), "phase_boundary")
            todo_history.append((idx, len(items)))
        elif name in WRITE_EDIT_TOOLS and todo_history and not first_edit_after_plan_done:
            if idx > todo_history[-1][0]:
                add(messages[idx].get("g"), "first_edit_after_plan")
                first_edit_after_plan_done = True

    # 测试类
    if verification_assistants:
        add(messages[verification_assistants[0]].get("g"), "first_test_run")
        last_test = verification_assistants[-1]
        prev_assistant = max(
            (i for i, m in enumerate(messages[:last_test]) if m.get("role") == "assistant"),
            default=None,
        )
        if prev_assistant is not None:
            add(messages[prev_assistant].get("g"), "before_last_test_run")

    # 完成宣告与错误修复
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if COMPLETION_PATTERN.search(content):
            add(message.get("g"), "completion_claim")
        calls_here = message.get("tool_calls") or []
        if any(c.get("function", {}).get("name") in WRITE_EDIT_TOOLS + ("bash",) for c in calls_here):
            # 上一条工具结果 is_error 或含失败信息
            prev_tools = [
                m for m in messages[:idx] if m.get("role") == "tool"
            ]
            if prev_tools:
                last_tool = prev_tools[-1]
                if last_tool.get("is_error") or re.search(r"failed|error|AssertionError", last_tool.get("content") or "", re.I):
                    add(message.get("g"), "error_to_fix")

    if len(triggers) >= 3:
        return sorted(triggers, key=lambda t: t.g)

    # 备选触发点：token_spike 与 reread_cluster
    for idx, message in enumerate(messages):
        if message.get("role") == "tool" and count_text_tokens(message.get("content") or "") > 8000:
            later = next(
                (i for i in range(idx + 1, len(messages)) if messages[i].get("role") == "assistant"),
                None,
            )
            if later is not None:
                add(messages[later].get("g"), "token_spike")
    read_records: list[tuple[int, str]] = []
    for idx, call in calls:
        if call.get("function", {}).get("name") == "read_file":
            read_records.append((idx, _parse_args(call).get("path", "")))
    for i, (idx_i, path_i) in enumerate(read_records):
        cluster = [idx_i]
        for idx_j, path_j in read_records[i + 1 :]:
            if idx_j - idx_i <= 5 and path_j == path_i:
                cluster.append(idx_j)
        if len(cluster) >= 3:
            later = next(
                (i for i in range(cluster[-1] + 1, len(messages)) if messages[i].get("role") == "assistant"),
                cluster[-1],
            )
            add(messages[later].get("g"), "reread_cluster")
    return sorted(triggers, key=lambda t: t.g)


def allocate_budget(n: int, char_len: int, max_injections: int = MAX_INJECTIONS) -> dict[str, int]:
    """调和平均式预算：base = max(1, floor(n*S/(n+S)))，长轨迹加成，间隔 max(2, floor(n/(e+1)))。"""
    base = max(1, math.floor(n * max_injections / (n + max_injections)))
    if char_len > 100_000:
        base += 2
    elif char_len > 50_000:
        base += 1
    e = min(max_injections, base)
    interval = max(2, math.floor(n / (e + 1)))
    return {"base": base, "e": e, "interval": interval}


def choose_trigger_points(triggers: list[Trigger], char_len: int) -> list[Trigger]:
    """完成宣告必选（不占预算），其余按优先级与最小间隔贪心选取。"""
    mandatory = [t for t in triggers if t.type == "completion_claim"]
    mandatory_g = {t.g for t in mandatory}
    # 同一条消息只注入一次：完成宣告所在消息不再作为其他类型的候选
    candidates = [t for t in triggers if t.type != "completion_claim" and t.g not in mandatory_g]
    budget = allocate_budget(len(candidates), char_len)
    e, interval = budget["e"], budget["interval"]
    # 最小间隔按候选点的时间顺序（序号）计；选取顺序按类型优先级
    chronological = sorted(candidates, key=lambda t: t.g)
    position = {id(t): pos for pos, t in enumerate(chronological)}
    order = {name: rank for rank, name in enumerate(PRIORITY)}
    by_priority = sorted(candidates, key=lambda t: (order.get(t.type, len(PRIORITY)), t.g))
    chosen: list[Trigger] = []
    chosen_positions: list[int] = []
    chosen_g: set[int] = set()
    for trigger in by_priority:
        if len(chosen) >= e:
            break
        pos = position[id(trigger)]
        if trigger.g in chosen_g:
            continue
        if any(abs(pos - other) < interval for other in chosen_positions):
            continue
        chosen.append(trigger)
        chosen_positions.append(pos)
        chosen_g.add(trigger.g)
    return sorted(mandatory + chosen, key=lambda t: t.g)


def pass_quality_gate(text: str, trigger_type: str) -> bool:
    """纯规则质量门：词数、状态词、完成宣告的验证依据、无祈使句。"""
    words = len(text.split())
    if words < MIN_REVIEW_WORDS or words > MAX_REVIEW_WORDS:
        return False
    lowered = text.lower()
    if not any(phrase in lowered for phrase in STATE_PHRASES):
        return False
    if trigger_type == "completion_claim" and not any(w in lowered for w in VERIFICATION_WORDS):
        return False
    if IMPERATIVE_PATTERN.search(text):
        return False
    return True


def generate_review(
    prefix_messages: list[dict[str, Any]],
    next_actions: list[str],
    client: Any,
    model: str,
) -> str:
    """调用注入模型生成英文状态回顾（temperature 0.6）。"""
    from trajectory_tools.prompts.injection import INJECTION_PROMPT

    history = "\n\n".join(
        f"[{m.get('role')}]\n{m.get('content') or ''}"
        + (
            "\n工具调用：" + "; ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')[:200]})"
                for c in m.get("tool_calls") or []
            )
            if m.get("tool_calls")
            else ""
        )
        for m in prefix_messages
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INJECTION_PROMPT},
            {
                "role": "user",
                "content": history + "\n\n接下来实际发生的三个动作：\n" + "\n".join(next_actions),
            },
        ],
        temperature=0.6,
    )
    return (response.choices[0].message.content or "").strip()


def apply_reviews(traj: dict, reviews: dict[int, str]) -> Optional[dict]:
    """把通过质量门的回顾写回 assistant 消息（前置拼接），meta.injections 追加。

    至少写入一个 completion_claim 或测试相关注入点才产出注入版；否则返回 None。
    """
    new_traj = json.loads(json.dumps(traj))
    injections = []
    by_g = {m.get("g"): m for m in new_traj["messages"]}
    for g, review in sorted(reviews.items()):
        message = by_g.get(g)
        if message is None or message.get("role") != "assistant":
            continue
        message["content"] = review + "\n\n" + (message.get("content") or "")
        injections.append({"g": g})
    if not injections:
        return None
    new_traj.setdefault("meta", {})["injections"] = injections
    return new_traj


def inject_trajectory(
    traj: dict,
    client: Any,
    model: str,
    prefix_token_budget: int = 60_000,
    recent_token_keep: int = 40_000,
) -> Optional[dict]:
    """对单条轨迹执行完整注入流程；不适合注入（无成功信号）返回 None。"""
    messages = traj["messages"]
    triggers = find_trigger_points(messages)
    chosen = choose_trigger_points(triggers, sum(len(m.get("content") or "") for m in messages))
    if not chosen:
        return None
    reviews: dict[int, str] = {}
    for trigger in chosen:
        idx = next(i for i, m in enumerate(messages) if m.get("g") == trigger.g)
        prefix = messages[:idx]
        # 超 60K token 时保留开头任务与最近 40K
        total = sum(count_text_tokens(m.get("content") or "") for m in prefix)
        if total > prefix_token_budget:
            head = prefix[:2]
            tail_budget = recent_token_keep
            tail = []
            for m in reversed(prefix[2:]):
                cost = count_text_tokens(m.get("content") or "")
                if tail_budget - cost < 0:
                    break
                tail_budget -= cost
                tail.append(m)
            prefix = head + list(reversed(tail))
        next_actions = []
        for m in messages[idx : idx + 4]:
            for call in m.get("tool_calls") or []:
                args = _parse_args(call)
                detail = args.get("path") or args.get("command") or ""
                next_actions.append(f"{call['function']['name']}({str(detail)[:100]})")
            if len(next_actions) >= 3:
                break
        review = generate_review(prefix, next_actions[:3], client, model)
        if pass_quality_gate(review, trigger.type):
            reviews[trigger.g] = review
    result = apply_reviews(traj, reviews)
    if result is None:
        return None
    # 记录注入类型
    by_g = {t.g: t.type for t in chosen}
    for entry in result["meta"]["injections"]:
        entry["type"] = by_g.get(entry["g"])
    has_strong = any(
        entry["type"] == "completion_claim" or entry["type"] in TEST_RELATED_TRIGGERS
        for entry in result["meta"]["injections"]
    )
    return result if has_strong else None


__all__ = []
