"""打包测试：配比计算、ms-swift 格式转换、截断后配对保持。"""

import json

from trajectory_tools.pair_validation import validate_pairs
from training.build_dataset import build_mix, to_swift_sample, trim_to_tokens, count_sample_tokens


def _traj(tid, n_rounds=2):
    """构造 n 轮调用的轨迹。"""
    messages = [
        {"g": 0, "role": "system", "content": "sys"},
        {"g": 1, "role": "user", "content": "task"},
    ]
    g = 2
    for i in range(n_rounds):
        messages.append(
            {
                "g": g,
                "role": "assistant",
                "content": f"step {i}",
                "reasoning": "thinking",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            }
        )
        g += 1
        messages.append({"g": g, "role": "tool", "tool_call_id": f"c{i}", "content": "out", "is_error": False})
        g += 1
    return {"id": tid, "instance_id": "tinydb", "messages": messages, "meta": {}}


def test_to_swift_sample_format():
    traj = _traj("t1")
    sample = to_swift_sample(traj, tools=[{"type": "function", "function": {"name": "bash"}}])
    assert sample["tools"][0]["function"]["name"] == "bash"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assistant = sample["messages"][2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    assert "g" not in assistant and "reasoning" not in assistant and "is_error" not in sample["messages"][3]
    assert validate_pairs(sample["messages"]) == []


def test_build_mix_ratio():
    success = [_traj(f"s{i}") for i in range(12)]
    injected = [_traj(f"i{i}") for i in range(4)]
    compacted = [_traj(f"k{i}") for i in range(2)]
    # N=12, mix=0.25 → 目标增强 4 条；3:1 → 3 注入 + 1 压缩
    result = build_mix(success, injected, compacted, mix=0.25, inject_to_compact="3:1")
    assert len(result["chosen"]) == 12
    assert len(result["augmented"]) == 4
    assert abs(result["actual_mix"] - 0.25) < 1e-9
    # 注入/压缩不足时按实际数量
    result = build_mix(success, injected[:2], compacted[:0], mix=0.25, inject_to_compact="3:1")
    assert len(result["augmented"]) == 2
    assert result["actual_mix"] == 2 / 14
    # baseline_raw：只用成功轨迹
    result = build_mix(success, injected, compacted, baseline_raw=True)
    assert result["augmented"] == [] and len(result["chosen"]) == 12
    # repeat_success：增强部分换成等量成功轨迹
    result = build_mix(success, [], [], mix=0.25, repeat_success=True)
    assert len(result["augmented"]) == 4
    assert all(t["id"].startswith("s") for t in result["augmented"])


def test_trim_keeps_pairs():
    traj = _traj("t1", n_rounds=6)
    sample = to_swift_sample(traj)
    original_tokens = count_sample_tokens(sample)
    # 把每轮内容撑大，确保需要截断
    for m in sample["messages"]:
        m["content"] = m["content"] + "x" * 3000
    big_tokens = count_sample_tokens(sample)
    assert big_tokens > original_tokens
    limit = big_tokens - 10
    trimmed = trim_to_tokens(sample, limit, tokenizer=None)
    assert trimmed is not None
    assert count_sample_tokens(trimmed) <= limit + 3000  # 粗估按消息整块减
    assert validate_pairs(trimmed["messages"]) == []
    # system 与任务保留
    assert trimmed["messages"][0]["role"] == "system"
    assert trimmed["messages"][1]["role"] == "user"


def test_trim_drops_unfixable():
    traj = _traj("t1", n_rounds=2)
    sample = to_swift_sample(traj)
    for m in sample["messages"]:
        m["content"] = "x" * 100
    # max_tokens 极小：截完全部轮次仍超 → None
    assert trim_to_tokens(sample, 10, tokenizer=None) is None
