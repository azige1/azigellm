"""为仓库批量运行 PyCG 构建调用图。"""

import os
import json
from pathlib import Path

from codeharvest.core import RepoRecord
from codeharvest.workspace import REPOS_DIR, GRAPHS_DIR
from codeharvest.extraction.config import RepoJobConfig
from codeharvest.parallel import run_jobs_parallel_iter
from codeharvest.analysis.callgraph import CallGraphBuilder, CallGraphNormalizer

def build_pycg_callgraph(repo: RepoRecord):
    cgraph = CallGraphBuilder.build_repo_call_graph(repo.repo_path)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(repo.callgraph_path, "w") as f:
        json.dump(cgraph, f, indent=4)

    CallGraphNormalizer.drop_unresolvable_callers(repo)
    CallGraphNormalizer.normalize_callee_identifiers(repo)

    id2type = CallGraphNormalizer.build_id_kind_map(repo)
    id2type = {str(k): v.name for k, v in id2type.items()}
    cgdict = repo.callgraph.to_dict()

    new_cgraph = {"graph": cgdict, "id2type": id2type}

    with open(repo.callgraph_path, "w") as f:
        json.dump(new_cgraph, f, indent=4)

    return new_cgraph

def run_pycg_for_repos(repo_args: RepoJobConfig):
    all_repos_clones: list[str] = sorted(os.listdir(REPOS_DIR))
    all_repos = [
        RepoRecord.from_file_path(REPOS_DIR / repo_clone_name)
        for repo_clone_name in all_repos_clones
    ]
    print(f"Running pycg on {len(all_repos)} repos")
    if repo_args.pycg_multiprocess == 0:
        for repo in all_repos:
            if not Path(repo.callgraph_path).exists():
                build_pycg_callgraph(repo)
    else:
        outputs = run_jobs_parallel_iter(
            build_pycg_callgraph,
            all_repos,
            num_workers=repo_args.pycg_multiprocess,
            timeout_per_task=repo_args.pycg_timeout * 60,
            use_progress_bar=True,
            max_mem=8 * 1024 * 1024 * 1024,
        )

        for repo_id, output in zip(all_repos, outputs):
            if output.is_success():
                pass
            else:
                print(f"Failed to run pycg on {repo_id}: {output.exception_tb}")
                continue
