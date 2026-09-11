"""任务数据质量审计：对生成的终端任务做静态与容器级质量检查。

三个子模块共享同一份数据模型，均可独立导入：

- ``static_checks``     —— 纯静态检查（AST 结构分、题面重合度、模式完整性）
- ``container_checks``  —— 容器构建与初始/终态测试的真实执行
- ``solution_checks``   —— 解题结果（solution.json / run 级 result.json）检查

每个 ``check_task`` 返回 ``TaskAuditResult``；CLI 见仓库根目录的
``audit_tasks.py``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Decision(str, Enum):
    """审计结论。优先级：reject > flag > pass。"""

    PASS = "pass"
    FLAG = "flag"
    REJECT = "reject"


@dataclass
class TaskAuditResult:
    """单个任务的审计结果。

    metrics 存数值型指标（如 structural_score），flags 存人类可读的
    命中原因（如 ``"structural_score_low"``）。
    """

    task_id: str
    task_dir: Path
    decision: Decision
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def worst_decision(results: list[TaskAuditResult]) -> Decision:
    """取一组结果中最严重的结论。"""
    if any(r.decision == Decision.REJECT for r in results):
        return Decision.REJECT
    if any(r.decision == Decision.FLAG for r in results):
        return Decision.FLAG
    return Decision.PASS
