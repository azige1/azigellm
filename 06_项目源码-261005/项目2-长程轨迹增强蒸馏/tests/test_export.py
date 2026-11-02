"""导出测试：同 id 聚合、全局索引、四类配对失败被识别。"""

import json

from trajectory_tools.export import merge_events, assign_global_index, export_trajectory
from trajectory_tools.pair_validation import (
    ERROR_DANGLING_CALL,
    ERROR_MISSING_RESULT,
    ERROR_ORPHAN_RESULT,
    ERROR_ORDER_VIOLATION,
    validate_pairs,
)


def _init():
    return {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "model": "m",
        "session_id": "s",
        "system_prompt": "sys",
        "task": "task",
    }


def _assistant_text(msg_id, text):
    return {
        "type": "assistant",
        "message": {"id": msg_id, "role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _assistant_call(msg_id, call_id, name="bash", args=None):
    return {
        "type": "assistant",
        "message": {
            "id": msg_id,
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": name, "input": args or {"command": "ls"}}
            ],
        },
    }


def _tool_result(call_id, content="ok", is_error=False):
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}
            ],
        },
    }


def _write_events(tmp_path, records):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    return path


def test_merge_same_id_blocks(tmp_path):
    records = [
        _init(),
        _assistant_text("a1", "先看"),
        _assistant_call("a1", "c1"),
        _tool_result("c1"),
        _assistant_text("a2", "TASK COMPLETE"),
    ]
    messages = merge_events(records)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "task"}
    a1 = messages[2]
    assert a1["role"] == "assistant"
    assert a1["content"] == "先看"
    assert len(a1["tool_calls"]) == 1 and a1["tool_calls"][0]["id"] == "c1"
    assert messages[3]["role"] == "tool" and messages[3]["tool_call_id"] == "c1"


def test_global_index():
    messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
    assign_global_index(messages)
    assert [m["g"] for m in messages] == [0, 1, 2]


def test_valid_trajectory_passes(tmp_path):
    path = _write_events(
        tmp_path,
        [_init(), _assistant_call("a1", "c1"), _tool_result("c1"), _assistant_text("a2", "done")],
    )
    traj, errors = export_trajectory(path)
    assert errors == []
    assert traj["meta"]["num_messages"] == 5


def test_dangling_call_detected():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
    ]
    assert ERROR_DANGLING_CALL in validate_pairs(messages)


def test_orphan_result_detected():
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
    assert ERROR_ORPHAN_RESULT in validate_pairs(messages)


def test_order_violation_detected():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "assistant", "content": "next"},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    assert ERROR_ORDER_VIOLATION in validate_pairs(messages)


def test_duplicate_call_id_detected():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
    ]
    assert ERROR_MISSING_RESULT in validate_pairs(messages)


def test_export_drops_invalid(tmp_path):
    path = _write_events(tmp_path, [_init(), _assistant_call("a1", "c1")])  # 无结果 → 悬空调用
    traj, errors = export_trajectory(path)
    assert traj is None
    assert ERROR_DANGLING_CALL in errors
