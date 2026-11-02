"""HOST 后端测试：在临时 git 仓库上执行简单脚本并回收文件（只依赖 git 与 bash）。"""

import subprocess
from pathlib import Path

import pytest

from rebuild_spec.core.backends import HostContext
from rebuild_spec.core.constants import CopyFiles
from rebuild_spec.core.specs import build_spec


def _make_task_repo(tmp_path: Path) -> Path:
    """造一个带两个提交的最小任务仓库。"""
    repo_dir = tmp_path / "origin" / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "reference"], cwd=repo_dir, check=True)
    reference = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo_dir / "pkg.py").write_text("VALUE = 1\n\ndef stub():\n    raise NotImplementedError\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "skeleton"], cwd=repo_dir, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()
    repo_dir.parent.joinpath("commits.json").write_text(
        __import__("json").dumps({"base": base, "reference": reference}), encoding="utf-8"
    )
    return repo_dir


def _make_spec(tmp_path: Path, repo_dir: Path) -> tuple[object, Path]:
    commits = __import__("json").loads((repo_dir.parent / "commits.json").read_text(encoding="utf-8"))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    instance = {
        "instance_id": "minrepo",
        "repo": "local/minrepo",
        "base_commit": commits["base"],
        "reference_commit": commits["reference"],
        "setup": {"python": "3.11"},
        "test": {"test_cmd": "python -m pytest tests", "test_dir": "tests"},
        "src_dir": ".",
        "local_path": str(repo_dir),
        "host_venv": str(repo_dir.parent / ".venv"),
    }
    spec = build_spec(instance, "scratch", absolute=False, repo_directory=str(work_dir / "repo"))
    return spec, work_dir


def test_host_context_runs_and_collects(tmp_path, monkeypatch):
    pytest.importorskip("git")
    repo_dir = _make_task_repo(tmp_path)
    spec, work_dir = _make_spec(tmp_path, repo_dir)
    # 测试环境不依赖 uv：跳过 venv 准备
    monkeypatch.setattr(HostContext, "_prepare_venv", lambda self: None)

    import logging

    logger = logging.getLogger("test-host")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    script = work_dir / "eval.sh"
    script.write_text("echo ok\n")
    files_to_copy = CopyFiles(
        eval_script={"src": script, "dest": Path("eval.sh")},
        patch={"src": script, "dest": Path("patch.diff")},
    )
    with HostContext(
        spec, logger, timeout=60, num_cpus=1, log_dir=log_dir,
        files_to_copy=files_to_copy, files_to_collect=["result.txt"],
    ) as ctx:
        output, timed_out, _ = ctx.exec_run_with_timeout("echo hello > repo/result.txt && bash eval.sh")
        assert not timed_out
        assert "ok" in output
    assert (log_dir / "result.txt").read_text(encoding="utf-8").strip() == "hello"
    # 临时克隆已清理，源仓库不受影响
    assert not (work_dir / "repo").exists()
    assert (repo_dir / "pkg.py").exists()
