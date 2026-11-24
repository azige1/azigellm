"""压缩恢复测试：切点顺延、尾部轮数、自指词质量门、B 段配对校验。"""

from trajectory_tools.compaction import (
    RECOVERY_GUIDE,
    assemble_part_b,
    find_cut_point,
    pass_quality_gate,
)


def _assistant(command=None, name="bash", g=None):
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{g}",
                "type": "function",
                "function": {"name": name, "arguments": '{"command": "%s"}' % command if command else "{}"},
            }
        ] if name else [],
    }
    if g is not None:
        msg["g"] = g
    return msg


def _tool(g):
    return {"role": "tool", "tool_call_id": f"c{g}", "content": "out"}


def _messages(n_turns, command="ls"):
    """构造 n_turns 轮 bash 消息对。"""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_turns):
        messages.append(_assistant(command, g=2 + i * 2))
        messages.append(_tool(2 + i * 2))
    return messages


def test_cut_point_deferred_to_read_or_test_turn():
    # 前 3 轮 ls（每轮内容 ~1000 字符），阈值 2000：crossed 后第一轮是 ls，应顺延到 pytest 轮
    messages = _messages(6, command="ls")
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = "x" * 1000
    idx = find_cut_point(messages, cut_threshold=2000)
    call = messages[idx]["tool_calls"][0]
    import json

    assert "pytest" in json.loads(call["function"]["arguments"])["command"]
    # 构造一轮 read_file：应优先停在 read_file 轮
    messages = _messages(6, command="ls")
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = "x" * 1000
    messages[8] = _assistant(None, name="read_file", g=8)
    idx = find_cut_point(messages, cut_threshold=2000)
    assert messages[idx]["tool_calls"][0]["function"]["name"] == "read_file"


def test_cut_point_requires_tail_length():
    messages = _messages(3, command="pytest tests")
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = "x" * 1000
    # 只有 3 轮，阈值 500：crossed 在第二轮，后面不足 4 条 → 不切
    assert find_cut_point(messages, cut_threshold=500) is None


def test_cut_point_no_crossing():
    messages = _messages(2, command="pytest tests")
    assert find_cut_point(messages, cut_threshold=10**9) is None


def test_quality_gate_banned_words():
    assert pass_quality_gate("Everything is on track and verified by tests.")
    assert not pass_quality_gate("Here is a summary of the session so far.")
    assert not pass_quality_gate("Compacting context to stay under the context limit.")
    assert pass_quality_gate("This session is being continued from a previous conversation.")


def test_recovery_guide_mentions_confirm():
    lowered = RECOVERY_GUIDE.lower()
    assert any(w in lowered for w in ("confirm", "re-read", "run the test"))


def test_assemble_part_b_keeps_g_and_validates():
    messages = [
        {"role": "system", "content": "sys", "g": 0},
        {"role": "user", "content": "task", "g": 1},
        {"role": "assistant", "content": "old work", "g": 2},
        {
            "role": "assistant",
            "content": "cut here",
            "g": 3,
            "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest tests"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "1 passed", "g": 4},
        {"role": "assistant", "content": "TASK COMPLETE", "g": 5},
    ]
    traj = {"id": "t1", "instance_id": "tinydb", "messages": messages, "meta": {"hidden_pass_rate": 1.0}}
    part_b = assemble_part_b(traj, 3, "Nine sections of factual review.")
    assert part_b is not None
    assert part_b["meta"]["segment"] == {"part": "B", "of": 2, "cut_token": part_b["meta"]["segment"]["cut_token"]}
    tail_gs = [m["g"] for m in part_b["messages"][-3:]]
    assert tail_gs == [3, 4, 5]
    assert part_b["messages"][2]["content"].startswith("This session is being continued")


def test_assemble_part_b_rejects_invalid():
    messages = [
        {"role": "system", "content": "s", "g": 0},
        {"role": "user", "content": "t", "g": 1},
        {"role": "assistant", "content": "cut", "g": 2, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ]},
    ]
    traj = {"id": "t", "instance_id": "x", "messages": messages, "meta": {}}
    assert assemble_part_b(traj, 2, "review") is None  # 悬空调用 → 校验失败
