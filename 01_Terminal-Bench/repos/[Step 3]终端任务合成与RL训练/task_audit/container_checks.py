"""容器级审计：真实构建任务镜像并执行初始 / 终态测试。

本模块复用 ``task_factory.docker_env_gen.build_and_verify_image``，
不重复实现 Docker 构建逻辑，只在其之上补充结构化的结果判定与
批次级异常信号。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from task_audit import Decision, TaskAuditResult


# 批次级异常阈值
BUILD_FAILURE_RATE_WARN = 0.3
TIMEOUT_RATE_WARN = 0.1
TRIVIAL_SOLVED_RATE_WARN = 0.05


def _build_and_verify(*args, **kwargs):
    """延迟导入构建函数：只有真正做容器检查时才加载 docker 相关依赖。"""
    from task_factory.docker_env_gen import build_and_verify_image

    return build_and_verify_image(*args, **kwargs)


def check_task(task_dir: Path) -> TaskAuditResult:
    """对单个任务执行容器级检查：构建镜像、跑初始测试、验证终态测试会失败。"""
    task_dir = Path(task_dir)
    flags: list[str] = []
    metrics: dict[str, Any] = {}

    dockerfile_path = task_dir / "environment" / "Dockerfile"
    init_path = task_dir / "environment" / "test_initial_state.py"
    final_path = task_dir / "tests" / "test_final_state.py"

    for p in (dockerfile_path, init_path):
        if not p.exists():
            return TaskAuditResult(
                task_dir.name, task_dir, Decision.REJECT,
                metrics, [f"missing_{p.name}"],
            )

    dockerfile_text = dockerfile_path.read_text(encoding="utf-8", errors="replace")
    init_src = init_path.read_text(encoding="utf-8", errors="replace")
    final_src = (
        final_path.read_text(encoding="utf-8", errors="replace")
        if final_path.exists() else None
    )

    ok, output = _build_and_verify(dockerfile_text, init_src, final_test_py=final_src)

    lower = output.lower()
    metrics["build_and_test_ok"] = ok

    if ok:
        return TaskAuditResult(task_dir.name, task_dir, Decision.PASS, metrics, flags)

    # 按失败输出归因：结构性语法错误与"不做事也能通过"拒绝；超时与 OOM 标记
    if output.startswith("DOCKERFILE_PARSE_ERROR:"):
        decision, reason = Decision.REJECT, "dockerfile_parse_error"
    elif "trivially solved" in lower:
        decision, reason = Decision.REJECT, "trivially_solved"
    elif "timed out" in lower or "timeout" in lower:
        decision, reason = Decision.FLAG, "container_timeout"
    elif "out of memory" in lower or "oom" in lower or "exit 137" in lower:
        decision, reason = Decision.FLAG, "container_oom"
    else:
        decision, reason = Decision.REJECT, "build_or_initial_test_failed"

    flags.append(reason)
    metrics["error_excerpt"] = output[:500]
    return TaskAuditResult(task_dir.name, task_dir, decision, metrics, flags)


def check_batch(task_dirs: list[Path], max_workers: int = 4) -> list[TaskAuditResult]:
    """并行执行容器级检查，返回与输入对齐的结果列表。"""
    results: list[TaskAuditResult | None] = [None] * len(task_dirs)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        future_to_idx = {pool.submit(check_task, d): i for i, d in enumerate(task_dirs)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]


def batch_anomalies(results: list[TaskAuditResult]) -> dict[str, Any]:
    """批次级异常信号：用于发现生成模型 / prompt 的系统性故障。

    这些信号只进报告，不影响单任务结论。
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "build_failure_rate": 0.0,
            "timeout_rate": 0.0,
            "trivially_solved_rate": 0.0,
            "warnings": [],
        }

    n_failed = sum(
        1 for r in results
        if r.decision == Decision.REJECT
        and any(f in ("build_or_initial_test_failed", "dockerfile_parse_error") for f in r.flags)
    )
    n_timeout = sum(1 for r in results if "container_timeout" in r.flags)
    n_trivial = sum(1 for r in results if "trivially_solved" in r.flags)

    rates = {
        "build_failure_rate": round(n_failed / total, 4),
        "timeout_rate": round(n_timeout / total, 4),
        "trivially_solved_rate": round(n_trivial / total, 4),
    }

    warnings = []
    if rates["build_failure_rate"] > BUILD_FAILURE_RATE_WARN:
        warnings.append("high_build_failure_rate")
    if rates["timeout_rate"] > TIMEOUT_RATE_WARN:
        warnings.append("high_timeout_rate")
    if rates["trivially_solved_rate"] > TRIVIAL_SOLVED_RATE_WARN:
        warnings.append("high_trivially_solved_rate")

    return {"total": total, **rates, "warnings": warnings}
