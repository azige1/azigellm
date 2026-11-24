"""抽取流水线：遍历所有仓库，输出函数与方法清单。"""

import fire

from codeharvest.core import RepoRecord
from codeharvest.utils.io import dump_callable_records
from codeharvest.extraction.config import RepoJobConfig
from codeharvest.workspace import REPOS_DIR, EXTRACTION_DIR
from codeharvest.parallel import run_jobs_parallel_iter
from codeharvest.extraction.targets.repo_data import extract_repo_targets

def extract_repo_callables(repo_args: RepoJobConfig):
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    extraction_path = EXTRACTION_DIR / f"{repo_args.exp_id}_extracted.json"
    if extraction_path.exists():
        if repo_args.overwrite_extracted:
            print("Overwriting existing functions and methods. Interrupt to cancel!")
        else:
            print(
                "Extracted file already exists. Use --overwrite_extracted to overwrite existing."
            )
            return

    repo_dirs = list(REPOS_DIR.glob("*"))
    repos = [(RepoRecord.from_file_path(str(repo_dir)), repo_args) for repo_dir in repo_dirs]

    functions = []
    methods = []

    outputs = run_jobs_parallel_iter(
        extract_repo_targets,
        repos,
        num_workers=repo_args.extraction_multiprocess,
        use_progress_bar=True,
        progress_bar_desc="Extracting..",
    )

    for output in outputs:
        if output.is_success():
            new_functions, new_methods = output.result  
            functions.extend(new_functions)
            methods.extend(new_methods)
        else:
            print(f"Error extracting repo data: {output.exception_tb}")

    print(f"Extracted {len(functions)} functions and {len(methods)} methods")

    dump_callable_records(functions + methods, extraction_path)

if __name__ == "__main__":
    repo_args = fire.Fire(RepoJobConfig)
    extract_repo_callables(repo_args)
