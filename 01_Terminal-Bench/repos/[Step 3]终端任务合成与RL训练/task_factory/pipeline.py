"""任务合成流水线（Apptainer/SIF 路线）。

按批执行五个阶段：任务草稿 → 初始测试 → 终态测试 → Apptainer 定义 → 落盘。
每个阶段都是一次批量 LLM 调用 + 本地校验，任一阶段失败的条目被丢弃。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid
import asyncio

from tqdm import tqdm


from task_factory.template_gen import generate_task_specs_batch
from task_factory.initial_test_gen import generate_initial_tests_batch
from task_factory.apptainer_env_gen import batch_generate_def_files
from task_factory.final_test_gen import generate_final_tests_batch



@dataclass
class GenerationConfig:
    num_tasks: int
    out_dir: Path
    max_def_retries: int = 5
    max_num_completions: int = 4
    num_solutions: int = 256
    max_actions: int = 20
    model: str = "qwen/qwen-3-32b"
    max_tokens: int = 1024
    task_temperature: float = 1.0
    test_temperature: float = 0.6
    solution_temperature: float = 1.0
    parallel_jobs: int = 1
    verbose: bool = False


def write_text(path: Path, content: str) -> None:
    """先建父目录再写文本，避免目录不存在导致的报错。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_sif(def_path: Path, sif_path: Path) -> bool:
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.run(["apptainer", "build", str(sif_path), str(def_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        return rc == 0
    except FileNotFoundError:
        # apptainer 未安装或不在 PATH
        return False
    except subprocess.TimeoutExpired:
        return False

def _format_task_dir(base: Path, idx: int, width: int = 6) -> Path:
    suffix = uuid.uuid4().hex[:8]
    return base / f"task_{idx:0{width}d}_{suffix}"

def _write_task_bundle(
    task_dir: Path,
    task_obj: Dict[str, Any],
    initial_test_code: str,
    def_text: str,
    final_test_code: str,
    summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    task_json = task_dir / "task.json"
    init_py = task_dir / "test_initial_state.py"
    final_py = task_dir / "test_final_state.py"
    def_file = task_dir / "container.def"
    sif_file = task_dir / "container.sif"
    sol_dir = task_dir / "solutions"
    sol_dir.mkdir(parents=True, exist_ok=True)

    write_text(task_json, json.dumps(task_obj, indent=4))
    write_text(init_py, initial_test_code)
    write_text(final_py, final_test_code)
    write_text(def_file, def_text)
    write_text(sol_dir / "summary.json", json.dumps(summary, indent=4))

    return task_json, init_py, final_py, def_file, sif_file


@dataclass
class BatchGenerationConfig(GenerationConfig):
    batch_size: int = 64
    max_concurrency: int = 64



def _generate_task_batch(cfg: BatchGenerationConfig, batch_count: int) -> List[Optional[Path]]:

    # 1) 任务草稿
    print(f"Generating {batch_count} task templates with {cfg.max_concurrency} concurrency")
    task_templates = generate_task_specs_batch(
        batch_count,
        model=cfg.model,
        temperature=cfg.task_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    if not task_templates:
        print("No task templates generated")
        return []

    descriptions: List[str] = [t.get("description", "").strip() for t in task_templates]
    truths: List[str] = [t.get("truth", "").strip() for t in task_templates]


    # 提前剔除空描述 / 空 truth
    valid_indices = [i for i, (d, tr) in enumerate(zip(descriptions, truths)) if d and tr]
    if not valid_indices:
        print("No valid task templates generated")
        return []


    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]

    print(f"Task templates generated: {len(descriptions)}")

    # 2) 初始测试
    print(f"Generating {len(descriptions)} initial tests with {cfg.max_concurrency} concurrency")
    init_tests = generate_initial_tests_batch(
        list(zip(descriptions, truths)),
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    valid_indices = [i for i, test in enumerate(init_tests) if test]
    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]

    print(f"Generated {len(init_tests)} initial tests")

    # 3) 终态测试
    print(f"Generating {len(descriptions)} final tests with {cfg.max_concurrency} concurrency")
    final_tests = generate_final_tests_batch(
        list(zip(descriptions, truths, init_tests)),
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    print(f"Generated {len(final_tests)} final tests")
    valid_indices = [i for i, test in enumerate(final_tests) if test]
    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]
    final_tests = [final_tests[i] for i in valid_indices]


    # 4) Apptainer 定义：每条一次生成，随后本地构建验证
    print(f"Generating {len(descriptions)} defs with {cfg.max_concurrency} concurrency")
    def_candidates = batch_generate_def_files(
        list(zip(descriptions, truths, init_tests)),
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=min(64, cfg.max_concurrency),
    )

    valid_indices = [i for i, def_text in enumerate(def_candidates) if def_text]
    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]
    final_tests = [final_tests[i] for i in valid_indices]
    def_candidates = [def_candidates[i] for i in valid_indices]

    print(f"Generated {len(def_candidates)} defs")
    # 5) 落盘成功条目
    print(f"Saving {len(descriptions)} tasks")
    saved_paths: List[Optional[Path]] = []
    for i in range(len(descriptions)):
        desc = descriptions[i]
        tr = truths[i]
        init_py = init_tests[i]
        def_text = def_candidates[i]
        final_py = final_tests[i]

        if not desc or not tr or not init_py or not def_text or not final_py:
            saved_paths.append(None)
            continue

        task_dir = _format_task_dir(cfg.out_dir, idx=0)  # idx 不参与命名，uuid 保证唯一
        task_obj = {"description": desc, "truth": tr, "name": task_dir.name}

        task_json, init_path, final_path, def_path, sif_path = _write_task_bundle(
            task_dir, task_obj, init_py, def_text, final_py, summary={}
        )

        saved_paths.append(task_dir)

    return saved_paths


def run_generation_pipeline(cfg: BatchGenerationConfig) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 按批凑够 num_tasks；因各阶段存在失败率，实际成功数可能更少
    requested = cfg.num_tasks
    batch_size = max(1, cfg.batch_size)

    all_saved: List[Optional[Path]] = []
    remaining = requested

    for _ in tqdm(range((requested + batch_size - 1) // batch_size)):
        count = min(batch_size, remaining)
        results = _generate_task_batch(cfg, count)
        all_saved.extend(results)
        remaining -= count

    saved = [p for p in all_saved if p is not None]
    summary = {
        "requested": requested,
        "succeeded": len(saved),
        "success_rate": (len(saved) / requested) if requested else 0.0,
        "saved_dirs": [str(p) for p in saved],
    }
    return summary


def parse_args(argv: Optional[List[str]] = None) -> BatchGenerationConfig:
    ap = argparse.ArgumentParser(description="Generate tasks via async-batched LLM calls.")
    ap.add_argument("--num-tasks", type=int, default=100, help="How many tasks to request")
    ap.add_argument("--out-dir", type=Path, default=Path("tasks"), help="Output directory")
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--task-temperature", type=float, default=1.0)
    ap.add_argument("--test-temperature", type=float, default=0.6)
    ap.add_argument("--solution-temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=128)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)
    verbose = args.verbose and not args.quiet


    return BatchGenerationConfig(
        num_tasks=args.num_tasks,
        out_dir=args.out_dir,
        model=args.model,
        task_temperature=args.task_temperature,
        test_temperature=args.test_temperature,
        solution_temperature=args.solution_temperature,
        parallel_jobs=1,
        verbose=verbose,
        batch_size=max(1, args.batch_size),
        max_concurrency=max(1, args.max_concurrency),
    )


if __name__ == "__main__":
    cfg = parse_args()
    summary = run_generation_pipeline(cfg)
    print(json.dumps(summary, indent=4))
