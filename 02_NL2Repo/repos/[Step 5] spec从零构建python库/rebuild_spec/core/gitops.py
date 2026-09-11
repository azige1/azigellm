"""git 操作、日志与文本处理的通用工具函数。"""

import git
import git.exc
import hashlib
import logging
import os
import time
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

from fastcore.net import HTTP404NotFoundError, HTTP403ForbiddenError  # type: ignore
from ghapi.core import GhApi


class EvaluationError(Exception):
    """评测流程出错时抛出，携带仓库名与日志位置。"""

    def __init__(self, repo: str, message: str, logger: logging.Logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.repo = repo
        self.log_file = ""
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.repo}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )


def make_file_logger(
    repo: str, log_file: Path, mode: str = "w", verbose: int = 1
) -> logging.Logger:
    """创建一个同时写日志文件（可选同步到 stdout）的 logger。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{repo}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if verbose == 2:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def release_logger(logger: logging.Logger) -> None:
    """关闭并摘除 logger 的所有 handler，避免句柄泄漏。"""
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def short_hash(input_string: str) -> str:
    """取字符串 SHA-256 摘要的前 22 位，作为短标识。"""
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode("utf-8"))
    return sha256.hexdigest()[:22]


def slice_test_output(ss: str, pattern: str) -> str:
    """从带 `+` 前缀的 shell 回显日志中截取某条命令的输出片段。"""
    s = ss.split("\n")
    out = []
    append = False
    for one in s:
        if one.startswith("+") and pattern in one:
            append = True
        # 下一条命令开始，说明目标命令的输出已读完
        elif append and one.startswith("+"):
            out = out[1:]
            return "\n".join(out).strip()
        if append:
            out.append(one)
    return ""


def clone_repo(
    clone_url: str, clone_dir: str, branch: str, logger: logging.Logger
) -> git.Repo:
    """把仓库克隆到指定目录；若已存在则拉取更新，最后检出目标分支。

    参数
    ----
    clone_url : 仓库地址
    clone_dir : 本地目标目录
    branch    : 要检出的分支或提交
    logger    : 日志对象

    失败时抛出 RuntimeError。
    """
    if os.path.exists(clone_dir):
        logger.info(f"Repository already exists at {clone_dir}. Fetching updates.")
        try:
            repo = git.Repo(clone_dir)
            repo.git.fetch()
        except git.exc.GitCommandError as e:
            raise RuntimeError(f"Failed to fetch updates for repository: {e}")
    else:
        logger.info(f"Cloning {clone_url} into {clone_dir}")
        try:
            repo = git.Repo.clone_from(clone_url, clone_dir)
        except git.exc.GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")

    try:
        repo.git.checkout(branch)
    except git.exc.GitCommandError as e:
        raise RuntimeError(f"Failed to check out {branch}: {e}")

    return repo


def ensure_github_repo(
    organization: str, repo: str, logger: logging.Logger, token: Optional[str] = None
) -> None:
    """确保组织下存在同名远端仓库：存在则跳过，不存在则创建。

    遇到限流时按剩余额度轮询等待。
    """
    api = GhApi(token=token)
    while True:
        try:
            api.repos.get(owner=organization, repo=repo)  # type: ignore
            logger.info(f"{organization}/{repo} already exists")
            break
        except HTTP403ForbiddenError:
            while True:
                rl = api.rate_limit.get()  # type: ignore
                logger.info(
                    f"Rate limit exceeded for the current GitHub token,"
                    f"waiting for 5 minutes, remaining calls: {rl.resources.core.remaining}"
                )
                if rl.resources.core.remaining > 0:
                    break
                time.sleep(60 * 5)
        except HTTP404NotFoundError:
            api.repos.create_in_org(org=organization, name=repo)  # type: ignore
            logger.info(f"Created {organization}/{repo} on GitHub")
            break


def diff_between_commits(repo: git.Repo, old_commit: str, new_commit: str) -> str:
    """生成两个提交之间的 diff 文本（排除压缩后的 spec 文件）。"""
    try:
        patch = repo.git.diff(
            old_commit, new_commit, "--", ".", ":(exclude)spec.pdf.bz2"
        )
        return patch + "\n\n"
    except git.GitCommandError as e:
        raise Exception(f"Error generating patch: {e}")


def current_branch(repo_path: Union[str, Path]) -> str:
    """返回仓库当前所在分支名；游离 HEAD 时抛出异常并给出提示。"""
    repo = git.Repo(repo_path)
    try:
        branch = repo.active_branch.name
    except TypeError as e:
        raise Exception(
            f"{e}\nThis means the repository is in a detached HEAD state. "
            "To proceed, please specify a valid branch by using --branch {branch}."
        )

    return branch


def extract_python_blocks(text: str) -> List[str]:
    """提取文本中所有 ```python ... ``` 代码块的内容。"""
    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1).strip() for match in matches]


__all__ = []
