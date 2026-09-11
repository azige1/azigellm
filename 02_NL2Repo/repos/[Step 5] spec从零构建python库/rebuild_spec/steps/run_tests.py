"""在隔离环境里运行指定测试用例：生成分支 diff → 打补丁 → 跑 pytest → 回收报告。"""

import git
import os
import sys
import traceback
from datasets import load_dataset
from pathlib import Path

from typing import Iterator, Union
from rebuild_spec.core.constants import (
    EVAL_BACKENDS,
    CopyFiles,
    RUN_PYTEST_LOG_DIR,
    RepoRecord,
    FunctionRecord,
)
from rebuild_spec.core.specs import build_spec
from rebuild_spec.core.gitops import (
    EvaluationError,
    short_hash,
    diff_between_commits,
    make_file_logger,
    release_logger,
    extract_python_blocks,
)
from rebuild_spec.core.backends import (
    ExecBackend,
    DockerContext,
    ModalContext,
    E2BContext,
)
from rebuild_spec.steps.build_images import classify_dataset


def main(
    dataset_name: str,
    dataset_split: str,
    base_dir: str,
    repo_or_repo_dir: str,
    branch: str,
    test_ids: str,
    coverage: bool,
    backend: str,
    timeout: int,
    num_cpus: int,
    rebuild_image: bool,
    verbose: int,
) -> None:
    """运行某个仓库分支上的指定测试。

    测试可以跑在本地 Docker 容器里，也可以跑在云端沙箱里。
    """
    dataset: Iterator[Union[RepoRecord, FunctionRecord]] = load_dataset(
        dataset_name, split=dataset_split
    )  # type: ignore
    dataset_name = dataset_name.lower()
    absolute = backend != "e2b"
    spec = None
    example = None
    repo_name = None
    dataset_type = None
    for example in dataset:
        if repo_or_repo_dir.endswith("/"):
            repo_or_repo_dir = repo_or_repo_dir[:-1]
        dataset_type = classify_dataset(dataset_name)
        if dataset_type in ("swebench", "simple"):
            repo_name = example["instance_id"]
        else:
            repo_name = example["repo"].split("/")[-1]
        if repo_name in os.path.basename(repo_or_repo_dir) or repo_or_repo_dir.endswith(
            repo_name
        ):
            spec = build_spec(example, dataset_type, absolute)
            break
    assert spec is not None, "No spec available"
    assert example is not None, "No example available"
    assert repo_name is not None, "No repo available"
    assert dataset_type is not None, "No dataset_type available"

    hashed_test_ids = short_hash(test_ids)
    # 准备日志目录
    log_dir = RUN_PYTEST_LOG_DIR / repo_name / branch / hashed_test_ids
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_pytest.log"
    logger = make_file_logger(repo_name, log_file, verbose=verbose)

    if dataset_type != "simple":
        try:
            local_repo = git.Repo(repo_or_repo_dir)
            logger.info(f"Loaded a git repo from {repo_or_repo_dir}")
        except (git.exc.NoSuchPathError, git.exc.InvalidGitRepositoryError):  # type: ignore
            repo_dir = os.path.join(base_dir, repo_name)
            logger.error(
                f"{repo_or_repo_dir} is not a git dir, trying {repo_dir} again"
            )
            try:
                local_repo = git.Repo(repo_dir)
                logger.info(f"Retried succeeded. Loaded a git repo from {repo_dir}")
            except git.exc.NoSuchPathError:  # type: ignore
                raise Exception(
                    f"{repo_dir} and {repo_or_repo_dir} are not git directories.\nUsage: rebuild-spec test {{repo_dir}} {{branch}} {{test_ids}}"
                )
            except Exception as e:
                raise e
        commit_id = ""
        if branch == "reference":
            commit_id = example["reference_commit"]
        else:
            # 先查本地分支
            if branch in local_repo.branches:
                commit_id = local_repo.commit(branch).hexsha
            else:
                found_remote_branch = False
                for remote in local_repo.remotes:
                    remote.fetch()  # 先同步每个远端的最新引用

                    for ref in remote.refs:
                        if (
                            ref.remote_head == branch
                        ):  # 去掉远端前缀再比较分支名
                            commit_id = local_repo.commit(ref.name).hexsha
                            found_remote_branch = True
                            break
                    if found_remote_branch:
                        break
                if not found_remote_branch:
                    raise Exception(
                        f"Branch {branch} does not exist locally or remotely."
                    )

        # 生成补丁
        if "swe" in dataset_name:
            if branch == "reference":
                patch = (
                    example["test"]["patch"] + "\n\n" + example["test"]["test_patch"]
                )
            else:
                patch = diff_between_commits(
                    local_repo, example["base_commit"], commit_id
                )
                patch += "\n\n" + example["test"]["test_patch"]
        else:
            patch = diff_between_commits(
                local_repo, example["base_commit"], commit_id
            )

        # 生成评测脚本
        if coverage:
            coverage_text = (
                f" --cov={example['src_dir']} --cov-branch --cov-report json"
            )
        else:
            coverage_text = ""
        eval_script = spec.eval_script.format(test_ids=test_ids, coverage=coverage_text)

    else:
        if branch == "reference":
            patch = (
                example["prompt"]
                + "\n\n"
                + example["canonical_solution"]
                + "\n\n"
                + example["test"]
            )
        else:
            solution = test_ids
            prompt = example["prompt"] if "prompt" in example.keys() else ""
            matches = extract_python_blocks(solution)
            if len(matches) > 0:
                solution = "\n\n".join(matches)
            else:
                solution = prompt + "\n\n" + solution
            patch = solution + "\n\n" + example["test"]
        eval_script = spec.eval_script

    patch_file = Path(log_dir / "patch.diff")
    patch_file.write_text(patch, encoding="utf-8", errors="ignore")
    eval_file = Path(log_dir / "eval.sh")
    eval_file.write_text(eval_script)

    backend = backend.upper()
    if ExecBackend(backend) == ExecBackend.MODAL:
        logger.info("Running on Modal")
        execution_context = ModalContext
    elif ExecBackend(backend) == ExecBackend.LOCAL:
        logger.info("Running locally")
        execution_context = DockerContext
    elif ExecBackend(backend) == ExecBackend.E2B:
        logger.info("Running E2B")
        execution_context = E2BContext
    else:
        raise ValueError(
            f"Evaluation must be from {', '.join(EVAL_BACKENDS)}, but {backend} is provided."
        )

    files_to_copy = CopyFiles(
        eval_script={
            "src": eval_file,
            "dest": Path("/eval.sh" if absolute else "eval.sh"),
        },
        patch={
            "src": patch_file,
            "dest": Path("/patch.diff" if absolute else "patch.diff"),
        },
    )
    files_to_collect = [
        "report.json",
        "pytest_exit_code.txt",
        "test_output.txt",
    ]
    if coverage:
        files_to_collect.append("coverage.json")

    eval_command = (
        "/bin/bash /eval.sh"
        if ExecBackend(backend) != ExecBackend.E2B
        else "/bin/bash eval.sh"
    )
    try:
        with execution_context(
            spec,
            logger,
            timeout,
            num_cpus,
            log_dir,
            files_to_copy,
            files_to_collect,
            rebuild_image,
        ) as context:
            output, timed_out, total_runtime = context.exec_run_with_timeout(
                eval_command
            )
            logger.info(output)
            if timed_out:
                raise EvaluationError(
                    repo_name,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )
        release_logger(logger)
        if verbose > 0:
            test_output = Path(log_dir / "test_output.txt")
            print(test_output.read_text())
        pytest_exit_code = Path(log_dir / "pytest_exit_code.txt").read_text().strip()
        sys.exit(int(pytest_exit_code))
    except EvaluationError as e:
        error_msg = (
            f"Error in running pytest for {repo_name}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({log_file}) for more information."
        )
        raise EvaluationError(repo_name, error_msg, logger)
    except Exception as e:
        error_msg = (
            f"General error: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({log_file}) for more information."
        )
        raise RuntimeError(error_msg)


__all__ = []
