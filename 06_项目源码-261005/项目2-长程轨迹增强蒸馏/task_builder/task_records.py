"""任务记录：生成与 tasks/test.jsonl 的读写。"""

import json
from pathlib import Path
from typing import Optional


def make_record(
    name: str,
    github: str,
    base_commit: str,
    reference_commit: str,
    setup: dict,
    test_cmd: str,
    test_dir: str,
    src_dir: str,
    task_dir: Path,
    split: str,
    reference_pass_rate: Optional[float] = None,
) -> dict:
    """组装一条与 RepoRecord 一致的任务记录。"""
    record = {
        "instance_id": name,
        "repo": github,
        "base_commit": base_commit,
        "reference_commit": reference_commit,
        "setup": setup,
        "test": {"test_cmd": test_cmd, "test_dir": test_dir},
        "src_dir": src_dir,
        "local_path": str((task_dir / "repo").resolve()),
        "spec_path": str((task_dir / "spec.md").resolve()),
        "hidden_tests_dir": str((task_dir / "hidden_tests").resolve()),
        "host_venv": None,
        "split": split,
    }
    if reference_pass_rate is not None:
        record["reference_pass_rate"] = reference_pass_rate
    return record


def append_record(tasks_dir: Path, record: dict) -> None:
    """向 tasks/test.jsonl 追加一条记录。"""
    with (tasks_dir / "test.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(tasks_dir: Path) -> list[dict]:
    """读取全部任务记录。"""
    path = tasks_dir / "test.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rewrite_records(tasks_dir: Path, records: list[dict]) -> None:
    """整体重写 tasks/test.jsonl。"""
    with (tasks_dir / "test.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = []
