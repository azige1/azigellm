#!/usr/bin/env python3
"""任务质量审计命令行入口。

子命令：
    offline     静态检查（AST 结构分 / 题面重合度 / 模式完整性）
    online      容器级检查（真实构建镜像并跑测试，较慢）
    solutions   解题结果检查（基于 solution/solution.json）
    all         依次执行以上全部

示例：
    python audit_tasks.py offline --tasks-dir harbor_tasks --report audit_report.json
    python audit_tasks.py online  --tasks-dir harbor_tasks --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from task_audit import TaskAuditResult
from task_audit import static_checks, container_checks, solution_checks


def _list_task_dirs(tasks_dir: Path) -> list[Path]:
    """列出任务目录（目录名以 task 开头）。"""
    if not tasks_dir.exists():
        print(f"Error: tasks dir not found: {tasks_dir}", file=sys.stderr)
        raise SystemExit(1)
    return sorted(
        d for d in tasks_dir.iterdir()
        if d.is_dir() and d.name.startswith("task")
    )


def _print_table(results: list[TaskAuditResult]) -> None:
    """在终端打印简要的结论统计与最严重的若干条。"""
    n_pass = sum(1 for r in results if r.decision.value == "pass")
    n_flag = sum(1 for r in results if r.decision.value == "flag")
    n_reject = sum(1 for r in results if r.decision.value == "reject")
    print(f"\ntotal={len(results)}  pass={n_pass}  flag={n_flag}  reject={n_reject}")

    shown = 0
    for r in results:
        if r.decision.value == "reject" and shown < 20:
            print(f"  [reject] {r.task_id}: {', '.join(r.flags)}")
            shown += 1


def _write_report(
    report_path: Path,
    tasks_dir: Path,
    static_results: list[TaskAuditResult] | None,
    online_results: list[TaskAuditResult] | None,
    solution_results: list[TaskAuditResult] | None,
) -> None:
    """把各组件结果合并写入 JSON 报告。"""
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tasks_dir": str(tasks_dir),
    }

    if static_results is not None:
        report["summary"] = static_checks.dataset_summary(static_results)
        report["tasks"] = [
            {
                "task_id": r.task_id,
                "decision": r.decision.value,
                "metrics": {k: v for k, v in r.metrics.items() if k != "missing_files"},
                "flags": r.flags,
            }
            for r in static_results
        ]

    if online_results is not None:
        report["batch_anomalies"] = container_checks.batch_anomalies(online_results)
        report["online"] = [
            {"task_id": r.task_id, "decision": r.decision.value, "flags": r.flags}
            for r in online_results
        ]

    if solution_results is not None:
        report["solutions"] = [
            {
                "task_id": r.task_id,
                "decision": r.decision.value,
                "metrics": r.metrics,
                "flags": r.flags,
            }
            for r in solution_results
        ]

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["offline", "online", "solutions", "all"])
    parser.add_argument("--tasks-dir", type=Path, default=Path("harbor_tasks"))
    parser.add_argument("--report", type=Path, default=Path("audit_report.json"))
    parser.add_argument("--workers", type=int, default=4, help="online 模式的并发构建数")
    args = parser.parse_args()

    task_dirs = _list_task_dirs(args.tasks_dir)
    print(f"Auditing {len(task_dirs)} task dirs under {args.tasks_dir} (mode={args.mode})")

    static_results = online_results = solution_results = None

    if args.mode in ("offline", "all"):
        static_results = static_checks.check_batch(task_dirs)
        _print_table(static_results)

    if args.mode in ("online", "all"):
        online_results = container_checks.check_batch(task_dirs, max_workers=args.workers)
        _print_table(online_results)
        anomalies = container_checks.batch_anomalies(online_results)
        if anomalies["warnings"]:
            print(f"Batch anomalies: {anomalies['warnings']}")

    if args.mode in ("solutions", "all"):
        solution_results = solution_checks.check_batch(task_dirs)
        _print_table(solution_results)

    _write_report(args.report, args.tasks_dir, static_results, online_results, solution_results)


if __name__ == "__main__":
    main()
