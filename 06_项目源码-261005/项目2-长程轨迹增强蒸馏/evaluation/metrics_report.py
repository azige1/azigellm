"""行为指标汇总：按模型聚合 rollout 结果，行为指标与隐藏通过率分开报告。

指标：盲改段数、最长盲改、接口签名错位率、工具报错率、完成宣告前未执行
测试占比、压缩后首个动作（Read/测试）比例；隐藏通过率单列；并按训练前
学生是否命中 answer_access 把任务分两组报告得分变化。
"""

import json
from pathlib import Path
from typing import Any, Optional

from trajectory_tools.export import export_trajectory
from trajectory_tools.behavior_metrics import (
    completion_before_verification,
    compute_blind_metrics,
    first_action_after_compact,
    signature_mismatch_rate,
    tool_error_rate,
)
from trajectory_tools.attribution import attribute


def evaluate_run(
    run_dir: Path,
    task: dict,
) -> Optional[dict[str, Any]]:
    """评测单条 rollout：行为指标 + 隐藏通过率。"""
    events = run_dir / "events.jsonl"
    if not events.exists():
        return None
    traj, errors = export_trajectory(events)
    if traj is None:
        return None
    messages = traj["messages"]
    blind = compute_blind_metrics(messages)
    tool_stats = tool_error_rate(messages)
    score_file = run_dir / "score.json"
    pass_rate = None
    if score_file.exists():
        try:
            pass_rate = json.loads(score_file.read_text(encoding="utf-8")).get("hidden_pass_rate")
        except json.JSONDecodeError:
            pass_rate = None
    # answer_access 只看实际执行的 bash 命令，缺报告不影响判定
    modes = attribute(traj, task, run_dir / "report.json", run_dir / "test_output.txt")
    result: dict[str, Any] = {
        "id": f"{task['instance_id']}__{run_dir.name}",
        "instance_id": task["instance_id"],
        "blind_segments": blind["blind_segments"],
        "max_blind": blind["max_blind"],
        "tool_calls": tool_stats["tool_calls"],
        "tool_error_rate": tool_stats["error_rate"],
        "completion_before_verification": completion_before_verification(messages),
        "first_action_after_compact": first_action_after_compact(messages),
        "has_compact_boundary": traj["meta"]["has_compact_boundary"],
        "hidden_pass_rate": pass_rate,
        "answer_access": "answer_access" in modes,
        "signature_mismatch_rate": None,
    }
    workspace = run_dir / "workspace"
    if (workspace / "repo").is_dir():
        try:
            sig = signature_mismatch_rate(
                Path(task["local_path"]),
                workspace / "repo",
                task["src_dir"],
                task["base_commit"],
            )
            result["signature_mismatch_rate"] = sig["mismatch_rate"]
        except Exception:  # noqa: BLE001
            pass
    return result


def build_report(runs_dir: Path, tasks_file: Path) -> dict[str, Any]:
    """对 runs/ 下全部 rollout 聚合，返回报告字典。"""
    tasks = {}
    for line in Path(tasks_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            tasks[record["instance_id"]] = record
    results = []
    for instance_dir in sorted(Path(runs_dir).iterdir()):
        if not instance_dir.is_dir() or instance_dir.name not in tasks:
            continue
        for run_dir in sorted(instance_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            one = evaluate_run(run_dir, tasks[instance_dir.name])
            if one is not None:
                results.append(one)

    def mean(values: list[float]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    boundary_runs = [r for r in results if r["has_compact_boundary"]]
    first_actions = [r["first_action_after_compact"] for r in boundary_runs if r["first_action_after_compact"]]
    report = {
        "num_runs": len(results),
        "behavior_metrics": {
            "blind_segments": mean([r["blind_segments"] for r in results]),
            "max_blind": mean([r["max_blind"] for r in results]),
            "signature_mismatch_rate": mean([r["signature_mismatch_rate"] for r in results]),
            "tool_error_rate": mean([r["tool_error_rate"] for r in results]),
            "completion_before_verification_ratio": mean(
                [1.0 if r["completion_before_verification"] else 0.0 for r in results]
            ),
            "post_compact_read_or_test_ratio": mean(
                [1.0 if a in ("read", "test") else 0.0 for a in first_actions]
            ) if first_actions else None,
            "num_compact_boundary_runs": len(boundary_runs),
        },
        "hidden_pass_rate": mean([r["hidden_pass_rate"] for r in results]),
        "by_answer_access": {
            "hit": {
                "num_runs": sum(1 for r in results if r["answer_access"]),
                "hidden_pass_rate": mean(
                    [r["hidden_pass_rate"] for r in results if r["answer_access"]]
                ),
            },
            "miss": {
                "num_runs": sum(1 for r in results if not r["answer_access"]),
                "hidden_pass_rate": mean(
                    [r["hidden_pass_rate"] for r in results if not r["answer_access"]]
                ),
            },
        },
        "runs": results,
    }
    return report


def report_markdown(report: dict[str, Any], model_name: str) -> str:
    """把报告渲染成 Markdown 表。"""
    lines = [
        f"# 评测结果：{model_name}",
        "",
        f"rollout 次数：{report['num_runs']}",
        "",
        "## 行为指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    for key, value in report["behavior_metrics"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## 隐藏测试通过率",
        "",
        f"平均：{report['hidden_pass_rate']}",
        "",
        "| 分组（answer_access） | 次数 | 平均通过率 |",
        "| --- | --- | --- |",
    ]
    for group, data in report["by_answer_access"].items():
        lines.append(f"| {group} | {data['num_runs']} | {data['hidden_pass_rate']} |")
    return "\n".join(lines) + "\n"


__all__ = []
