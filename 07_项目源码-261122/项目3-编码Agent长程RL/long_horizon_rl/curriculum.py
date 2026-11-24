"""两阶段 curriculum 与题池筛选（3.1 节）。

- 阶段一：项目一函数级任务（最多 8 轮，工具集 Bash / Read / Edit）；
- 阶段二：项目二仓库级任务（最多 60 轮 / 参考配置 120 轮，完整 7 工具集）；
- 开训前筛选：用当前 checkpoint 测 pass@8，仅保留 1/8 至 6/8 的题目；
- 阶段二动态清理：每 30 步重测剩余题库，将全部通过的题目移出题池。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class PassAt8Entry:
    """单道题目的 pass@8 筛选结果。"""

    task_id: str
    passed: int
    total: int
    pass_rates: list[float]
    kept: bool


def should_keep_pass_at_8(
    passed_count: int,
    total_samples: int = 8,
    min_passed: int = 1,
    max_passed: int = 6,
) -> bool:
    """判断 pass@k 是否落在 [min_passed, max_passed] 区间（默认 1/8 至 6/8）。"""
    if total_samples <= 0:
        return False
    return min_passed <= passed_count <= max_passed


def filter_tasks_by_pass_at_8(
    tasks: Iterable[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], list[float]],
    *,
    num_samples: int = 8,
    min_passed: int = 1,
    max_passed: int = 6,
    pass_threshold: float = 1.0 - 1e-6,
) -> tuple[list[dict[str, Any]], list[PassAt8Entry]]:
    """对题池逐题运行 ``num_samples`` 次采样判分，只保留通过次数在 ``[min_passed, max_passed]`` 的题目。"""
    kept_tasks: list[dict[str, Any]] = []
    report: list[PassAt8Entry] = []
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("instance_id") or "")
        rates = list(evaluator(task))[:num_samples]
        passed_count = sum(1 for r in rates if r >= pass_threshold)
        kept = should_keep_pass_at_8(
            passed_count,
            total_samples=len(rates) or num_samples,
            min_passed=min_passed,
            max_passed=max_passed,
        )
        entry = PassAt8Entry(
            task_id=task_id,
            passed=passed_count,
            total=len(rates),
            pass_rates=rates,
            kept=kept,
        )
        report.append(entry)
        if kept:
            enriched = dict(task)
            enriched["init_pass_at_8"] = f"{passed_count}/{len(rates)}"
            kept_tasks.append(enriched)
    return kept_tasks, report


def prune_fully_solved_tasks(
    tasks: Iterable[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], float],
    *,
    pass_threshold: float = 1.0 - 1e-6,
) -> tuple[list[dict[str, Any]], list[str]]:
    """阶段二每 30 步重测剩余题库，将全部测试通过的题目移出题池。"""
    remaining: list[dict[str, Any]] = []
    removed_ids: list[str] = []
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("instance_id") or "")
        rate = float(evaluator(task))
        if rate >= pass_threshold:
            removed_ids.append(task_id)
        else:
            remaining.append(task)
    return remaining, removed_ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out
