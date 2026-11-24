"""按仓库运行 pytest 与 examples 采集真实调用，输出按函数 id 聚合的 JSON。"""

import ast
import glob
import json
import os
import subprocess
import sys

from pydantic import BaseModel, Field

from codeharvest.utils.io import read_callable_records
from codeharvest.workspace import (
    REPOS_DIR,
    CAPTURED_CALLS_DIR,
    EXTRACTED_DATA_DIR,
)
from codeharvest.analysis.instrument.side_effect_filter import SideEffectFilter

INSTRUMENT_DIR = os.path.dirname(os.path.abspath(__file__))
STANDALONE_PLUGIN = os.path.join(INSTRUMENT_DIR, "standalone_capture.py")

EXAMPLE_SCRIPT_TIMEOUT = 60


class CaptureConfig(BaseModel):
    exp_id: str = Field("temp", description="实验 ID，决定抽取结果与输出文件名")
    in_file: str | None = Field(None, description="抽取结果文件名，缺省 {exp_id}_extracted.json")
    test_timeout: int = Field(240, description="单仓库 pytest 超时（秒）")
    capture_multiprocess: int = Field(1, description="并行采集的进程数，0 表示串行")
    run_examples: bool = Field(True, description="是否运行 examples/*.py 脚本采集")


def _repo_python(repo_path: str) -> str:
    from codeharvest.extraction.local_setup import repo_venv_python

    return repo_venv_python(repo_path) or sys.executable


def _module_name_from_relative_path(relative_path: str) -> str:
    if relative_path.endswith(".py"):
        relative_path = relative_path[:-3]
    parts = [p for p in relative_path.replace("\\", "/").split("/") if p != "__init__"]
    return ".".join(parts)


def _build_targets(functions, repo_path: str) -> list[dict]:
    """生成采集目标清单，静态副作用判定命中的函数不挂钩子。"""
    file_cache: dict[str, ast.Module] = {}

    def module_ast(file_path: str):
        if file_path not in file_cache:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                file_cache[file_path] = ast.parse(f.read())
        return file_cache[file_path]

    targets = []
    for func in functions:
        file_path = func.file_path
        try:
            tree = module_ast(file_path)
        except (SyntaxError, OSError):
            continue

        module_functions = SideEffectFilter.module_function_table(tree)
        node = None
        if getattr(func, "class_name", None):
            for class_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                if class_node.name == func.class_name:
                    node = next(
                        (
                            m
                            for m in class_node.body
                            if isinstance(m, ast.FunctionDef) and m.name == func.name
                        ),
                        None,
                    )
                    break
        else:
            node = next(
                (
                    n
                    for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == func.name
                ),
                None,
            )
        if node is None:
            continue
        if SideEffectFilter.has_side_effects(node, module_functions, tree):
            continue

        module_name = _module_name_from_relative_path(func.file.relative_file_path)
        targets.append(
            {
                "module": module_name,
                "name": func.name,
                "class_name": getattr(func, "class_name", None),
                "func_id": func.id,
            }
        )
    return targets


def _run_pytest_for_repo(repo_path: str, targets_file: str, output_path: str, timeout: int) -> None:
    python = _repo_python(repo_path)
    env = os.environ.copy()
    env["CODEHARVEST_CAPTURE_TARGETS"] = targets_file
    env["CODEHARVEST_CAPTURE_OUTPUT"] = output_path
    plugin_dir = os.path.dirname(STANDALONE_PLUGIN)
    env["PYTHONPATH"] = plugin_dir + os.pathsep + repo_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [python, "-m", "pytest", "-q", "-p", "standalone_capture"],
        cwd=repo_path,
        env=env,
        capture_output=True,
        timeout=timeout,
    )


def _run_examples_for_repo(repo_path: str, targets_file: str, output_path: str) -> dict:
    python = _repo_python(repo_path)
    env = os.environ.copy()
    env["CODEHARVEST_CAPTURE_TARGETS"] = targets_file
    env["CODEHARVEST_CAPTURE_REPO_ROOT"] = repo_path

    example_scripts = sorted(
        glob.glob(os.path.join(repo_path, "examples", "*.py"))
        + glob.glob(os.path.join(repo_path, "example", "*.py"))
    )

    merged: dict[str, list] = {}
    for script in example_scripts:
        try:
            subprocess.run(
                [python, STANDALONE_PLUGIN, "--script", script, "--output", output_path],
                cwd=repo_path,
                env=env,
                capture_output=True,
                timeout=EXAMPLE_SCRIPT_TIMEOUT,
            )
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    data = json.load(f)
                for func_id, records in data.items():
                    merged.setdefault(func_id, [])
                    for record in records:
                        key = json.dumps(record.get("caller_info"), sort_keys=True)
                        merged[func_id] = [
                            r
                            for r in merged[func_id]
                            if json.dumps(r.get("caller_info"), sort_keys=True) != key
                        ]
                        merged[func_id].append(record)
                    merged[func_id] = merged[func_id][-3:]
        except subprocess.TimeoutExpired:
            print(f"examples 脚本超时: {script}")
    return merged


def _merge_records(base: dict, extra: dict) -> dict:
    for func_id, records in extra.items():
        base.setdefault(func_id, [])
        base[func_id].extend(records)
        base[func_id] = base[func_id][-3:]
    return base


def capture_repo(args_tuple) -> str:
    """单仓库采集入口（可并行调用），返回 repo_id。"""
    repo_id, functions, config, out_dir = args_tuple
    repo_path = os.path.join(str(REPOS_DIR), repo_id)
    if not os.path.isdir(repo_path):
        return repo_id

    os.makedirs(out_dir, exist_ok=True)
    targets_file = os.path.join(out_dir, f"{repo_id}_targets.json")
    output_path = os.path.join(out_dir, f"{repo_id}.json")

    targets = _build_targets(functions, repo_path)
    if not targets:
        return repo_id
    with open(targets_file, "w") as f:
        json.dump(targets, f, indent=2)

    try:
        _run_pytest_for_repo(repo_path, targets_file, output_path, config.test_timeout)
    except subprocess.TimeoutExpired:
        print(f"pytest 超时，跳过仓库 {repo_id}")
        return repo_id
    except Exception as e:
        print(f"仓库 {repo_id} 采集失败: {e!r}")
        return repo_id

    collected: dict = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            collected = json.load(f)

    if config.run_examples:
        example_records = _run_examples_for_repo(repo_path, targets_file, output_path)
        collected = _merge_records(collected, example_records)

    with open(output_path, "w") as f:
        json.dump(collected, f, indent=2)
    return repo_id


def run_capture(config: CaptureConfig) -> None:
    """对抽取结果按仓库分组，逐个运行 pytest 与 examples 采集。"""
    in_file = config.in_file or f"{config.exp_id}_extracted.json"
    functions = read_callable_records(EXTRACTED_DATA_DIR / in_file)

    by_repo: dict[str, list] = {}
    for func in functions:
        by_repo.setdefault(func.repo_id, []).append(func)

    out_dir = str(CAPTURED_CALLS_DIR / config.exp_id)
    tasks = [(repo_id, fs, config, out_dir) for repo_id, fs in sorted(by_repo.items())]

    if config.capture_multiprocess and config.capture_multiprocess > 1:
        from codeharvest.parallel import run_jobs_parallel_iter

        for output in run_jobs_parallel_iter(
            capture_repo,
            tasks,
            num_workers=config.capture_multiprocess,
            timeout_per_task=config.test_timeout + 120,
            use_progress_bar=True,
            progress_bar_desc="Capturing calls",
        ):
            if not output.is_success():
                print(f"采集失败: {output.exception_tb}")
    else:
        for task in tasks:
            capture_repo(task)

    print(f"采集完成: {out_dir}")
