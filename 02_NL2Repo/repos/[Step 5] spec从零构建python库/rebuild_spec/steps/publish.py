"""最后一步：把本地分支推送到自己的 GitHub 组织下留档。"""

import logging
import os

import git

from datasets import load_dataset
from typing import Iterator
from rebuild_spec.core.constants import RepoRecord, SPLIT
from rebuild_spec.core.gitops import ensure_github_repo


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    dataset_name: str,
    dataset_split: str,
    repo_split: str,
    base_dir: str,
    owner: str,
    branch: str,
    github_token: str,
) -> None:
    if github_token is None:
        # 未显式传 token 时读环境变量
        github_token = os.environ.get("GITHUB_TOKEN")
    dataset: Iterator[RepoRecord] = load_dataset(dataset_name, split=dataset_split)  # type: ignore
    for example in dataset:
        repo_name = example["repo"].split("/")[-1]
        if "swe" in dataset_name.lower():
            if repo_split != "all" and repo_split not in example["instance_id"]:
                continue
        else:
            if repo_split != "all" and repo_name not in SPLIT[repo_split]:
                continue
        local_repo_path = f"{base_dir}/{repo_name}"
        github_repo_url = f"https://github.com/{owner}/{repo_name}.git"
        github_repo_url = github_repo_url.replace(
            "https://", f"https://x-access-token:{github_token}@"
        )

        if not os.path.exists(local_repo_path):
            raise OSError(f"{local_repo_path} does not exists")
        else:
            repo = git.Repo(local_repo_path)

        # 远端没有同名仓库就先创建
        ensure_github_repo(
            organization=owner, repo=repo_name, logger=logger, token=github_token
        )
        remote_name = "submission"
        if remote_name not in [remote.name for remote in repo.remotes]:
            repo.create_remote(remote_name, url=github_repo_url)
        else:
            logger.info(
                f"Remote {remote_name} already exists, replacing it with {github_repo_url}"
            )
            repo.remote(name=remote_name).set_url(github_repo_url)

        if branch in repo.heads:
            repo.git.checkout(branch)
        else:
            raise ValueError(f"The branch {branch} you want save does not exist.")

        # 有未提交的改动时先兜底提交
        if not repo.is_dirty(untracked_files=True):
            repo.git.add(A=True)
            repo.index.commit("auto generated code.")

        origin = repo.remote(name=remote_name)
        try:
            origin.push(refspec=f"{branch}:{branch}")
            logger.info(f"Pushed to {github_repo_url} on branch {branch}")
        except Exception as e:
            logger.error(f"Push {branch} to {owner}/{repo_name} fails.\n{str(e)}")
            continue


__all__ = []
