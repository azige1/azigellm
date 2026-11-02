"""rollout 判分：把工作区改动提交成分支，走 HOST 后端评测并写 score.json。"""

import json
import shutil
from pathlib import Path

import git

from rebuild_spec.core.constants import RUN_PYTEST_LOG_DIR
from rebuild_spec.core.gitops import short_hash
from rebuild_spec.core.report_parser import parse_report
from rebuild_spec.steps.run_tests import evaluate_branch


# 运行产物不进入补丁：二进制缓存会让 git apply 失败
EXCLUDED_PATHSPECS = (
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/.pytest_cache/**",
    ":(exclude,glob)**/*.egg-info/**",
    ":(exclude,glob)**/.venv/**",
)


def commit_run_branch(workspace: Path, run_id: str) -> str:
    """把工作区相对骨架的源码改动提交到分支 run-{run_id}。"""
    repo = git.Repo(workspace)
    branch = f"run-{run_id}"
    if branch not in repo.branches:
        repo.git.checkout("-b", branch)
    else:
        repo.git.checkout(branch)
    repo.git.add("-A", "--", ".", *EXCLUDED_PATHSPECS)
    if repo.is_dirty(index=True, working_tree=False):
        repo.index.commit(f"rollout run {run_id}")
    return branch


def judge_run(
    workspace: Path,
    task: dict,
    run_id: str,
    out_dir: Path,
    dataset_name: str,
    dataset_split: str = "test",
    base_dir: str = "",
    timeout: int = 1800,
) -> dict:
    """评测一次 rollout 的工作区，返回 score 字典并写 score.json。"""
    branch = commit_run_branch(workspace, run_id)
    test_ids_file = Path(task["local_path"]).parent / "test_ids.txt"
    test_ids = (
        [line for line in test_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if test_ids_file.exists()
        else []
    )
    # 评测命令本身已包含测试目录（test_cmd = python -m pytest {test_dir}），这里不再追加用例 ID：
    # 用例 ID 若拼进 shell 命令，数百个参数既冗长也容易被换行拆成多条命令；
    # 通过率按 test_ids.txt 中的用例在 report.json 里逐条对照计算。
    test_ids_arg = ""
    score: dict = {"hidden_pass_rate": 0.0, "num_passed": 0, "num_tests": len(test_ids), "collection_error": False}
    try:
        evaluate_branch(
            dataset_name,
            dataset_split,
            base_dir,
            str(workspace),
            branch,
            test_ids_arg,
            False,
            "host",
            timeout,
            1,
            False,
            0,
            instance_id=task["instance_id"],
        )
        repo_name = task["repo"].split("/")[-1]
        log_dir = RUN_PYTEST_LOG_DIR / repo_name / branch / short_hash(test_ids_arg)
        for fname in ("report.json", "test_output.txt"):
            src = log_dir / fname
            if src.exists():
                shutil.copyfile(src, out_dir / fname)
        summary = parse_report(log_dir / "report.json", test_ids)
        score["hidden_pass_rate"] = summary["pass_rate"]
        score["num_passed"] = summary["num_passed"]
        if not (log_dir / "report.json").exists():
            score["collection_error"] = True
    except Exception as e:  # noqa: BLE001
        score["collection_error"] = True
        score["error"] = str(e)[:500]
    (out_dir / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    return score


__all__ = []
