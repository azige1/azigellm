"""解题结果审计：基于 solution.json 与 run 级 result.json 的质量信号。

- 离线：读任务目录里的 ``solution/solution.json``（Harbor 评测回收后写入）。
- 在线：读一次 Harbor 评测运行的 run 级 ``result.json``，给出批次级告警。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_audit import Decision, TaskAuditResult


# run 级告警阈值
ERROR_RATE_WARN = 0.05
LOW_PASS_RATE_WARN = 0.1

# 「从未解出 + verifier 结构分低」联合判定的阈值
SUSPECT_STRUCTURAL_SCORE = 0.2


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def check_task(task_dir: Path, structural_score: float | None = None) -> TaskAuditResult:
    """单个任务的解题结果检查。

    参数:
        task_dir: 任务目录（应含 ``solution/solution.json``）。
        structural_score: 静态审计算出的结构分；缺省时现场计算。
    """
    task_dir = Path(task_dir)
    flags: list[str] = []
    metrics: dict[str, Any] = {}
    decision = Decision.PASS

    solution_path = task_dir / "solution" / "solution.json"
    if not solution_path.exists():
        # 没有评测数据：不算缺陷，只标记为「未评测」
        return TaskAuditResult(task_dir.name, task_dir, Decision.FLAG, metrics, ["unevaluated"])

    data = _read_json(solution_path)
    if data is None:
        return TaskAuditResult(task_dir.name, task_dir, Decision.FLAG, metrics, ["solution_json_unreadable"])

    num_runs = int(data.get("num_runs", 0))
    num_success = int(data.get("num_success", 0))
    pass_at_k = data.get("pass_at_k", {}) or {}
    pass_at_1 = float(pass_at_k.get("1", pass_at_k.get(1, 0.0)) or 0.0)

    metrics.update(num_runs=num_runs, num_success=num_success, pass_at_1=pass_at_1)

    if pass_at_1 == 0.0 and num_runs >= 2:
        decision = Decision.FLAG
        flags.append("never_solved")

    if num_success == 0:
        if structural_score is None:
            # 现场补算结构分
            from task_audit.static_checks import check_task as static_check_task

            structural_score = float(
                static_check_task(task_dir).metrics.get("structural_score", 0.0)
            )
        metrics["structural_score"] = structural_score
        if structural_score < SUSPECT_STRUCTURAL_SCORE:
            # 从未解出且 verifier 几乎不检查结果：verifier 很可能本身是错的
            decision = Decision.REJECT
            flags.append("verifier_likely_wrong")

    return TaskAuditResult(task_dir.name, task_dir, decision, metrics, flags)


def check_batch(task_dirs: list[Path]) -> list[TaskAuditResult]:
    """对一批任务目录执行解题结果检查。"""
    return [check_task(d) for d in task_dirs]


def check_run(result_json: Path) -> dict[str, Any]:
    """检查一次 Harbor 评测运行的 run 级 result.json，返回异常信号字典。"""
    data = _read_json(Path(result_json))
    if data is None:
        return {"ok": False, "warnings": ["result_json_unreadable"]}

    n_total = int(data.get("n_total_trials", 0) or 0)
    stats = data.get("stats", {}) or {}
    n_errors = int(stats.get("n_errors", 0) or 0)

    evals = stats.get("evals", {}) or {}
    eval_obj: dict[str, Any] = next(iter(evals.values()), {}) if evals else {}
    metrics_list = eval_obj.get("metrics", []) or []
    mean_reward = None
    for m in metrics_list:
        if "mean" in m:
            mean_reward = m["mean"]
            break

    warnings: list[str] = []
    error_rate = (n_errors / n_total) if n_total else 0.0
    if n_total and error_rate > ERROR_RATE_WARN:
        warnings.append("high_error_rate")
    if mean_reward is not None and mean_reward < LOW_PASS_RATE_WARN:
        warnings.append("low_overall_pass_rate")

    return {
        "ok": True,
        "n_total_trials": n_total,
        "n_errors": n_errors,
        "error_rate": round(error_rate, 4),
        "mean_reward": mean_reward,
        "warnings": warnings,
    }
