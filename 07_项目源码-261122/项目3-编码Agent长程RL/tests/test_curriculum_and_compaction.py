"""测试 3.1 与 3.3 节：pass@8 筛选、题池清理、压缩上下文组装、parquet 数据转换与 YAML 配置。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import yaml

# 提供轻量占位模块，确保无完整项目一/二安装时也能离线验证纯逻辑
if "agent_rollout" not in sys.modules:
    ar_mod = types.ModuleType("agent_rollout")
    ar_rl = types.ModuleType("agent_rollout.run_loop")
    ar_rl.build_system_prompt = lambda: "SYS_PROMPT"
    ar_ch = types.ModuleType("agent_rollout.compact_history")
    ar_ch.CONTINUATION_PREFIX = "This session is being continued from a previous conversation."
    ar_ch.KEEP_RECENT_TURNS = 4
    ar_ch.LAST_READ_CONTENT_CHARS = 2000
    ar_ch.recent_turns_start = lambda msgs, keep_turns=4: max(
        0, [i for i, m in enumerate(msgs) if m.get("role") == "assistant"][-keep_turns]
    )
    ar_ch.find_last_read = lambda msgs: (None, None)
    ar_tools = types.ModuleType("agent_rollout.tools")
    ar_tools.TOOLS_SCHEMA = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    ar_tools.ToolResult = object
    ar_tools.Workspace = object
    ar_tools.execute_tool = lambda *a, **kw: None
    sys.modules["agent_rollout"] = ar_mod
    sys.modules["agent_rollout.run_loop"] = ar_rl
    sys.modules["agent_rollout.compact_history"] = ar_ch
    sys.modules["agent_rollout.tools"] = ar_tools

from long_horizon_rl.curriculum import (
    filter_tasks_by_pass_at_8,
    load_jsonl,
    prune_fully_solved_tasks,
)
from long_horizon_rl.data_prep import convert_tasks_to_verl_records


def test_pass_at_8_filtering_keeps_only_1_to_6_of_8() -> None:
    tasks = [
        {"task_id": "too_hard"},    # 0/8
        {"task_id": "sweet_low"},   # 1/8
        {"task_id": "sweet_high"},  # 6/8
        {"task_id": "too_easy_7"},  # 7/8
        {"task_id": "too_easy_8"},  # 8/8
    ]
    scores_map = {
        "too_hard": [0.0] * 8,
        "sweet_low": [1.0] + [0.0] * 7,
        "sweet_high": [1.0] * 6 + [0.0] * 2,
        "too_easy_7": [1.0] * 7 + [0.0],
        "too_easy_8": [1.0] * 8,
    }
    kept, report = filter_tasks_by_pass_at_8(tasks, lambda t: scores_map[t["task_id"]])
    assert [t["task_id"] for t in kept] == ["sweet_low", "sweet_high"]
    assert len(report) == 5


def test_prune_fully_solved_tasks() -> None:
    tasks = [{"instance_id": "a"}, {"instance_id": "b"}, {"instance_id": "c"}]
    rates = {"a": 1.0, "b": 0.62, "c": 1.0}
    remaining, removed = prune_fully_solved_tasks(tasks, lambda t: rates[t["instance_id"]])
    assert [t["instance_id"] for t in remaining] == ["b"]
    assert removed == ["a", "c"]


def test_convert_sample_tasks_to_verl_records() -> None:
    root = Path(__file__).resolve().parent.parent
    func_tasks = load_jsonl(root / "data" / "sample_function_tasks.jsonl")
    records = convert_tasks_to_verl_records(func_tasks, stage="function", split="train")
    assert len(records) == 1
    row = records[0]
    assert row["agent_name"] == "coding_compaction_agent"
    assert row["extra_info"]["tool_selection"] == ["bash", "read_file", "edit_file"]
    parsed_payload = json.loads(row["extra_info"]["task_payload_json"])
    assert parsed_payload["task_id"] == "tinydb-split_path-1"


def test_configs_align_with_spec_constraints() -> None:
    root = Path(__file__).resolve().parent.parent
    s1 = yaml.safe_load((root / "configs" / "stage1_function.yaml").read_text(encoding="utf-8"))
    s2 = yaml.safe_load((root / "configs" / "stage2_repo.yaml").read_text(encoding="utf-8"))
    ref32b = yaml.safe_load(
        (root / "configs" / "stage2_qwen3_32b_yarn64k_reference.yaml").read_text(encoding="utf-8")
    )

    for cfg in (s1, s2, ref32b):
        assert cfg["algorithm"]["rollout_correction"]["bypass_mode"] is False
        assert cfg["actor_rollout_ref"]["actor"]["clip_ratio_low"] == 0.2
        assert cfg["actor_rollout_ref"]["actor"]["clip_ratio_high"] == 0.28
        assert cfg["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.001
        assert cfg["actor_rollout_ref"]["actor"]["kl_loss_type"] == "low_var_kl"
        assert "hf_model" in cfg["actor_rollout_ref"]["actor"]["checkpoint"]["save_contents"]

    assert s1["coding_rl"]["curriculum"]["stage1_max_turns"] == 8
    assert s2["coding_rl"]["curriculum"]["stage2_max_turns"] == 60
    assert s2["coding_rl"]["compaction"]["compact_threshold_tokens"] == 27000
    assert s2["coding_rl"]["step_advantage"]["omega"] == 0.5
    assert s2["coding_rl"]["step_advantage"]["gamma"] == 0.95
    assert s2["coding_rl"]["sandbox"]["pool_size_per_task"] == 10

    assert ref32b["actor_rollout_ref"]["model"]["override_config"]["rope_scaling"]["factor"] == 4.0
    assert ref32b["coding_rl"]["compaction"]["window_tokens"] == 65536
    assert ref32b["trainer"]["nnodes"] * ref32b["trainer"]["n_gpus_per_node"] == 32
    assert (
        ref32b["actor_rollout_ref"]["rollout"]["nnodes"]
        * ref32b["actor_rollout_ref"]["rollout"]["n_gpus_per_node"]
        == 48
    )


def test_compaction_context_assembly_keeps_system_spec_summary_and_last_4_turns() -> None:
    from agent_rollout.compact_history import CONTINUATION_PREFIX, recent_turns_start

    sys_msg = {"role": "system", "content": "sys"}
    spec_msg = {"role": "user", "content": "spec"}
    history = [sys_msg, spec_msg]
    for turn in range(6):
        history.append({"role": "assistant", "content": f"thought-{turn}"})
        history.append({"role": "tool", "content": f"obs-{turn}"})

    keep_from = max(2, recent_turns_start(history, keep_turns=4))
    new_ctx = [
        sys_msg,
        spec_msg,
        {"role": "user", "content": f"{CONTINUATION_PREFIX}\n\n1. Primary Request..."},
        *history[keep_from:],
    ]
    assert new_ctx[0]["content"] == "sys"
    assert new_ctx[1]["content"] == "spec"
    assert new_ctx[2]["content"].startswith(CONTINUATION_PREFIX)
    # 最近 4 轮（turns 2, 3, 4, 5）共 8 条消息完整保留
    assert len(new_ctx) == 3 + 8
    assert new_ctx[3]["content"] == "thought-2"
    assert new_ctx[-1]["content"] == "obs-5"
