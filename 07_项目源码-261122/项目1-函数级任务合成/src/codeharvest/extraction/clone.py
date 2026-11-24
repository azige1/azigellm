"""从 URL 克隆或从本地复制仓库。"""

import os
import git
import json
import shutil
from pathlib import Path

import fire

from codeharvest.workspace import REPOS_DIR
from codeharvest.extraction.callgraph_pycg import run_pycg_for_repos
from codeharvest.extraction.config import RepoJobConfig
from codeharvest.parallel import run_jobs_parallel_iter

class RepoFetcher:

    @staticmethod
    def fetch_and_prepare(repo_args: RepoJobConfig):
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        if repo_args.repo_url:
            RepoFetcher.clone_from_url(repo_args.repo_url)

        elif repo_args.local_repo_path:
            RepoFetcher.copy_local_repo(repo_args.local_repo_path)

        elif repo_args.repo_paths_file:
            with open(repo_args.repo_paths_file) as f:
                repo_paths: list[str] = json.load(f)
                assert isinstance(
                    repo_paths, list
                ), f"Expected list, got {type(repo_paths)}"
                assert all(
                    isinstance(x, str) for x in repo_paths
                ), f"Expected list of strings, got {repo_paths}"
            RepoFetcher.copy_many_local(repo_paths, repo_args.cloning_multiprocess)

        elif repo_args.repo_urls_file:
            with open(repo_args.repo_urls_file) as f:
                repo_urls: list[str] = json.load(f)
                assert isinstance(
                    repo_urls, list
                ), f"Expected list, got {type(repo_urls)}"
                assert all(
                    isinstance(x, str) for x in repo_urls
                ), f"Expected list of strings, got {repo_urls}"
            RepoFetcher.clone_many_from_urls(repo_urls, repo_args.cloning_multiprocess)

        if repo_args.run_pycg_for_repos:
            run_pycg_for_repos(repo_args)

    @staticmethod
    def clone_from_url(repo_url: str):
        repo_username, repo_name = (
            repo_url.rstrip("/").removesuffix(".git").split("/")[-2:]
        )
        local_repo_clone_path = REPOS_DIR / f"{repo_username}___{repo_name}"

        if os.path.exists(local_repo_clone_path):
            print(
                f"Repository {repo_url} already exists at {local_repo_clone_path}... skipping"
            )
            return

        print(f"Cloning repository {repo_url} to {local_repo_clone_path}")
        git.Repo.clone_from(f"{repo_url}", local_repo_clone_path)

    @staticmethod
    def copy_local_repo(local_repo_path: str):
        
        local_repo_path = str(Path(local_repo_path).resolve())

        local_repo_name = local_repo_path.split("/")[-1]
        local_repo_clone_path = REPOS_DIR / f"LOCAL___{local_repo_name}"

        if os.path.exists(local_repo_clone_path):
            print(
                f"Repository {local_repo_path} already exists at {local_repo_clone_path}... skipping"
            )
            return

        print(f"Copying repository {local_repo_path} to {local_repo_clone_path}")

        shutil.copytree(local_repo_path, local_repo_clone_path)

    @staticmethod
    def clone_many_from_urls(repo_urls: list[str], cloning_multiprocess: int):
        if cloning_multiprocess > 0:
            output = run_jobs_parallel_iter(
                RepoFetcher.clone_from_url,
                repo_urls,
                cloning_multiprocess,
            )
            for result in output:
                if not result.is_success():
                    print(result.exception_tb)
        else:
            for repo_url in repo_urls:
                RepoFetcher.clone_from_url(repo_url)

    @staticmethod
    def copy_many_local(local_repo_paths: list[str], cloning_multiprocess: int):
        if cloning_multiprocess > 0:
            output = run_jobs_parallel_iter(
                RepoFetcher.copy_local_repo,
                local_repo_paths,
                cloning_multiprocess,
            )
            for result in output:
                if not result.is_success():
                    print(result.exception_tb)
        else:
            for local_repo_path in local_repo_paths:
                RepoFetcher.copy_local_repo(local_repo_path)

if __name__ == "__main__":
    repo_args = fire.Fire(RepoJobConfig)
    RepoFetcher.fetch_and_prepare(repo_args)
