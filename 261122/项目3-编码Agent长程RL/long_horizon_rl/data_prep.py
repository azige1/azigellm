"""将项目一函数级任务与项目二仓库级任务转换为 verl V1 parquet 数据集。

按通用约束：parquet 中异构嵌套字段序列化为 JSON 字符串，
并保持与项目二 SFT 一致的系统提示与工具集合。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets

from agent_rollout.run_loop import build_system_prompt

FUNCTION_TASK_SYSTEM_PROMPT = """你是一个 Python 编码助手。根据给出的函数签名与自然语言规格实现目标函数或方法。
你可以使用 bash、read_file、edit_file 工具查看代码并运行自测验证；完成后只回复最终实现代码或 TASK COMPLETE。"""


def build_function_prompt_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """构造阶段一函数级任务的 chat 消息列表。"""
    source_ref = task.get("source_ref") or {}
    user_text = (
        f"# 函数级编程任务\n\n"
        f"- 任务 ID: {task.get('task_id', '')}\n"
        f"- 目标文件: {source_ref.get('file_path', '')}\n\n"
        f"## 函数签名\n```python\n{task.get('signature', '')}\n```\n\n"
        f"## 规格说明\n{task.get('spec', '')}\n"
    )
    return [
        {"role": "system", "content": FUNCTION_TASK_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def build_repo_prompt_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """构造阶段二仓库级任务的 chat 消息列表（复用项目二系统提示）。"""
    spec_text = task.get("spec_text", "")
    if not spec_text and task.get("spec_path"):
        spec_path = Path(task["spec_path"])
        if spec_path.is_file():
            spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": spec_text},
    ]


def convert_tasks_to_verl_records(
    tasks: list[dict[str, Any]],
    *,
    stage: str,
    split: str = "train",
    stage1_tools: list[str] | None = None,
    stage2_tools: list[str] | None = None,
) -> list[dict[str, Any]]:
    """把项目一/二任务记录转成 verl ``RLHFDataset`` 兼容记录。"""
    s1_tools = stage1_tools or ["bash", "read_file", "edit_file"]
    s2_tools = stage2_tools or [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob_files",
        "grep_text",
        "todo_write",
    ]
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        if task.get("split") and task["split"] != split:
            continue
        if stage == "function":
            task_id = str(task.get("task_id", f"func-{idx}"))
            prompt = build_function_prompt_messages(task)
            tool_selection = s1_tools
            data_source = "codeharvest_function"
        elif stage == "repo":
            task_id = str(task.get("instance_id", f"repo-{idx}"))
            prompt = build_repo_prompt_messages(task)
            tool_selection = s2_tools
            data_source = "rebuild_spec_repo"
        else:
            raise ValueError(f"未知训练阶段: {stage}")

        # 异构嵌套字段统一存为 JSON 字符串，确保 Arrow/Parquet schema 稳定
        extra_info = {
            "index": idx,
            "task_id": task_id,
            "stage": stage,
            "split": split,
            "tool_selection": tool_selection,
            "task_payload_json": json.dumps(task, ensure_ascii=False),
        }
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": f"coding_{stage}",
                "agent_name": "coding_compaction_agent",
                "reward_model": {"style": "rule", "ground_truth": task_id},
                "extra_info": extra_info,
            }
        )
    return rows


def write_verl_parquet(rows: list[dict[str, Any]], out_parquet: Path) -> Path:
    """写出 verl 所需的 parquet 文件。"""
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(str(out))
    return out
