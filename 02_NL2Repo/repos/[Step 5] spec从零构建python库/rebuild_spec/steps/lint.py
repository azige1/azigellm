"""代码风格检查：用 pre-commit（ruff + pyright）检查指定文件或整个源码目录。"""

import subprocess
import sys
import os
from datasets import load_dataset
from pathlib import Path
from typing import Iterator, Union, List

from rebuild_spec.core.constants import RepoRecord


# 内置的 pre-commit 配置：常规检查 + ruff + pyright
HOOKS_CONFIG = """repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.3.0
  hooks:
  - id: check-case-conflict
  - id: mixed-line-ending

- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.6.1
  hooks:
    - id: ruff
      args: [ --fix ]
    - id: ruff-format

- repo: https://github.com/RobertCraigie/pyright-python
  rev: v1.1.376
  hooks:
    - id: pyright"""


def main(
    dataset_name: str,
    dataset_split: str,
    repo_or_repo_dir: str,
    files: Union[List[Path], None],
    base_dir: str,
) -> None:
    dataset: Iterator[RepoRecord] = load_dataset(dataset_name, split=dataset_split)  # type: ignore
    example = None
    repo_name = None
    for example in dataset:
        repo_name = example["repo"].split("/")[-1]
        if repo_or_repo_dir.endswith("/"):
            repo_or_repo_dir = repo_or_repo_dir[:-1]
        if repo_name in os.path.basename(repo_or_repo_dir):
            break
    assert example is not None, "No example available"
    assert repo_name is not None, "No repo available"

    if files is None:
        # 未指定文件时，检查源码目录下全部 .py 文件
        repo_dir = os.path.join(base_dir, repo_name)
        if os.path.isdir(repo_or_repo_dir):
            repo = repo_or_repo_dir
        elif os.path.isdir(repo_dir):
            repo = repo_dir
        else:
            raise Exception(
                f"Neither {repo_dir} nor {repo_or_repo_dir} is a valid path.\nUsage: rebuild-spec lint {{repo_or_repo_dir}}"
            )

        files = []
        repo = os.path.join(repo, example["src_dir"])
        for root, dirs, fs in os.walk(repo):
            for file in fs:
                if file.endswith(".py"):
                    files.append(Path(os.path.join(root, file)))

    config_file = Path(".rebuild-spec.pre-commit-config.yaml")
    if not config_file.is_file():
        config_file.write_text(HOOKS_CONFIG)
    command = ["pre-commit", "run", "--config", config_file, "--files"] + files
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(e.returncode)
    except FileNotFoundError:
        raise FileNotFoundError("Error: pre-commit command not found. Is it installed?")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


__all__ = []
