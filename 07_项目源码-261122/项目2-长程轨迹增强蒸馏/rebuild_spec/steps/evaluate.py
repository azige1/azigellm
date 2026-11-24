"""批量评估：对整个分组并发跑测试，汇总每个仓库的通过率与耗时。"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

from tqdm import tqdm
from pathlib import Path

from rebuild_spec.steps.run_tests import evaluate_branch
from rebuild_spec.steps.list_tests import main as get_tests
from rebuild_spec.core.constants import SPLIT, RUN_PYTEST_LOG_DIR
from rebuild_spec.core.dataset_loader import load_task_records
from rebuild_spec.core.gitops import short_hash, current_branch
from rebuild_spec.core.report_parser import parse_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    dataset_name: str,
    dataset_split: str,
    repo_split: str,
    base_dir: str,
    branch: Union[str, None],
    coverage: bool,
    backend: str,
    timeout: int,
    num_cpus: int,
    num_workers: int,
    rebuild_image: bool,
) -> None:
    dataset = list(load_task_records(dataset_name, dataset_split))
    is_local = any(one.get("local_path") for one in dataset)
    if "swe" in dataset_name.lower():
        if repo_split == "all":
            repos = [one["instance_id"] for one in dataset]
        else:
            repos = [
                one["instance_id"]
                for one in dataset
                if repo_split in one["instance_id"]
            ]
    elif is_local:
        # 本地任务：按记录自身的 split 字段过滤
        if repo_split == "all":
            repos = [one["instance_id"] for one in dataset]
        else:
            repos = [
                one["instance_id"]
                for one in dataset
                if one.get("split") == repo_split
            ]
    else:
        repos = SPLIT.get(repo_split, [])
    triples = []
    log_dirs = []
    task_dirs = {}
    for example in dataset:
        repo_name = example["repo"].split("/")[-1]
        if "swe" in dataset_name.lower():
            if repo_split != "all" and repo_split not in example["instance_id"]:
                continue
        elif is_local:
            if repo_split != "all" and example.get("split") != repo_split:
                continue
        else:
            if repo_split != "all" and repo_name not in SPLIT.get(repo_split, []):
                continue
        hashed_test_ids = short_hash(example["test"]["test_dir"])
        if branch is None:
            git_path = os.path.join(base_dir, example["instance_id"])
            branch = current_branch(git_path)
        log_dir = (
            RUN_PYTEST_LOG_DIR
            / example["instance_id"].split("/")[-1]
            / branch
            / hashed_test_ids
        )
        log_dirs.append(str(log_dir))
        triples.append((example["instance_id"], example["test"]["test_dir"], branch))
        if example.get("local_path"):
            task_dirs[example["instance_id"].split("/")[-1]] = str(
                Path(example["local_path"]).parent
            )

    with tqdm(total=len(repos), smoothing=0, desc="Evaluating repos") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_branch,
                    dataset_name,
                    dataset_split,
                    base_dir,
                    repo,
                    branch,
                    test_dir,
                    coverage,
                    backend,
                    timeout,
                    num_cpus,
                    rebuild_image,
                    0,
                ): None
                for repo, test_dir, branch in triples
            }
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    # 汇总结果
    out = []
    for name, example in zip(log_dirs, dataset):
        log_dir = name
        report_file = os.path.join(log_dir, "report.json")
        repo_name = example["instance_id"].split("/")[-1]
        test_ids = get_tests(
            repo_name, verbose=0, task_dir=task_dirs.get(repo_name)
        )
        test_ids = [xx for x in test_ids for xx in x if xx]
        summary = parse_report(report_file, test_ids)
        num_passed = summary["num_passed"]
        num_tests = summary["num_tests"]
        runtime = sum(
            call["duration"] for call in summary["outcomes"].values()
        )
        out.append(
            {
                "name": repo_name,
                "sum": runtime,
                "passed": summary["pass_rate"],
                "num_passed": num_passed,
                "num_tests": num_tests,
            }
        )
    print("repo,runtime,num_passed/num_tests")
    out = sorted(out, key=lambda x: x["sum"], reverse=True)
    for x in out:
        print(f"{x['name']},{x['sum']},{x['num_passed']}/{x['num_tests']}")
    averaged_passed = sum([x["passed"] for x in out]) / len(out) if out else 0.0
    print(f"average pass rate: {averaged_passed}")


__all__ = []
