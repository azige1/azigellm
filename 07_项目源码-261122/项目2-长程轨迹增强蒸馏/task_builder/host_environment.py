"""任务虚拟环境：建基准 venv、装依赖、导出锁文件、收集测试 ID 与参考通过率。"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rebuild_spec.core.report_parser import parse_report

PYTEST_PLUGINS = ["pytest", "pytest-cov", "pytest-json-report"]


def venv_env(venv: Path) -> dict:
    """在当前环境变量基础上激活 venv：保留原 PATH（uv 可能装在 conda 或 ~/.local/bin 下）。"""
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "")
    return env


def create_base_venv(task_dir: Path, python: str) -> Path:
    """在 tasks/{name}/.venv 建基准虚拟环境。"""
    venv = task_dir / ".venv"
    subprocess.run(["uv", "venv", "--python", python, str(venv)], check=True)
    return venv


def install_task_dependencies(venv: Path, setup: dict) -> None:
    """按 setup（packages / pip_packages）安装依赖与 pytest 插件。"""
    env = venv_env(venv)
    for key in ("packages", "pip_packages"):
        for one in setup.get(key) or []:
            subprocess.run(["uv", "pip", "install", one], check=True, env=env)
    subprocess.run(["uv", "pip", "install", *PYTEST_PLUGINS], check=True, env=env)


def write_requirements_lock(venv: Path, task_dir: Path, package_name: str) -> Path:
    """导出 requirements.lock，排除被测包自身。"""
    result = subprocess.run(
        ["uv", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        env=venv_env(venv),
    )
    normalized = package_name.lower().replace("-", "_")
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        and not line.startswith("-e ")
        and not line.split("==")[0].split(" @ ")[0].strip().lower().replace("-", "_").replace(".", "_")
        == normalized
    ]
    lock = task_dir / "requirements.lock"
    lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lock


def collect_test_ids(venv: Path, repo_dir: Path, test_dir: str) -> list[str]:
    """在参考实现上执行 pytest --collect-only，返回用例 ID 列表。"""
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-m", "pytest", "--collect-only", "-q", test_dir],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=900,
    )
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            ids.append(line)
        elif line.startswith("tests/") and line.endswith(".py"):
            # 模块级收集行
            ids.append(line)
    return ids


def run_reference_tests(
    venv: Path, repo_dir: Path, test_dir: str, test_ids: list[str]
) -> tuple[float, list[str], str]:
    """跑一遍参考实现，返回 (通过率, 在参考实现上通过的用例 ID, 原始输出)。

    判分只用参考实现能通过的用例：被跳过的（如缺 Redis 服务）与本身失败的用例
    不计入分母，否则满分达不到 1.0。
    """
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        result = subprocess.run(
            [
                str(venv / "bin" / "python"),
                "-m",
                "pytest",
                test_dir,
                "--json-report",
                f"--json-report-file={report_path}",
                "--continue-on-collection-errors",
            ],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        summary = parse_report(report_path, test_ids)
        passed_ids = [
            test_id
            for test_id in test_ids
            if summary["outcomes"].get(test_id, {}).get("outcome") == "passed"
        ]
        return summary["pass_rate"], passed_ids, result.stdout + result.stderr


__all__ = []
