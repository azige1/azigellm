"""训练数据准备：把任务目录转成 HF parquet（train/validation 两份）。

只保留「至少被解出过一次」的任务（按 solution/solution.json 的 pass@k
判断），可选地并行预构建各任务的 Docker 镜像。

用法：
    python -m rl.prepare_data --task-dir ./tasks --output-dir ./data --build-docker
"""
import subprocess
import sys
import pathlib
import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import datasets
import json
from pathlib import Path

sys.path.insert(0, str(pathlib.Path().resolve()))
from task_factory.rollout_agent import parse_agent_action, AGENT_SYSTEM_PROMPT, AGENT_USER_TEMPLATE


def build_task_image(task_dir_name, task_dir, verbose=True):
    """为单个任务构建 Docker 镜像，返回 ``(任务名, 是否成功)``。"""
    dockerfile = Path(task_dir) / task_dir_name / "environment" / "Dockerfile"

    if not dockerfile.exists():
        print(f"Dockerfile not found for {task_dir_name}")
        return task_dir_name, False

    tag = f"harbor-task-{task_dir_name}:latest"
    try:
        proc = subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(dockerfile), str(dockerfile.parent)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            if verbose:
                print(f"Docker build failed for {task_dir_name}: {proc.stderr[-200:]}")
            return task_dir_name, False
        return task_dir_name, True
    except Exception as e:
        print(f"Error building Docker image for {task_dir_name}: {e}")
        return task_dir_name, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--task-dir", default="./tasks")
    parser.add_argument("--difficulty", default="none")
    parser.add_argument("--max-time", default=300)
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--build-docker", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=20, help="Number of parallel workers for building containers")
    parser.add_argument(
        "--source", choices=["apptainer", "harbor"], default=None,
        help=(
            "任务过滤方式。'apptainer'：按 solutions/o3_summary.json 的 pass@16>0 过滤"
            "（SIF 路线任务）；'harbor'：按 solution/solution.json 的 num_success>0 过滤"
            "（Docker/Harbor 任务）。缺省时自动检测。"
        ),
    )

    args = parser.parse_args()
    random.seed(args.seed)
    task_dir_names = [f for f in os.listdir(args.task_dir) if "task" in f]
    # 只保留有解题结果且 pass@k 至少一项大于 0 的任务
    task_dir_names = [f for f in task_dir_names if (Path(args.task_dir) / f / "solution" / "solution.json").exists()]
    task_dir_names = [f for f in task_dir_names if any(v > 0 for v in json.load(open(Path(args.task_dir) / f / "solution" / "solution.json"))["pass_at_k"].values())]
    task_dir_names = list(sorted(task_dir_names))
    random.shuffle(task_dir_names)
    task_descriptions = [json.load(open(Path(args.task_dir) / f / "environment" / "task.json"))["description"] for f in task_dir_names]

    # 可选：并行预构建镜像
    failed_tasks = set()
    if args.build_docker:
        print(f"Building containers in parallel with {args.max_workers} workers...")
        completed = 0
        total = len(task_dir_names)
        progress_lock = Lock()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(build_task_image, task_dir_name, args.task_dir, verbose=False): task_dir_name
                for task_dir_name in task_dir_names
            }

            for future in as_completed(future_to_task):
                task_name, success = future.result()
                if not success:
                    failed_tasks.add(task_name)

                with progress_lock:
                    completed += 1
                    print(f"\rProgress: {completed}/{total} ({len(failed_tasks)} failed)", end='', flush=True)

        print()
        print(f"Container building complete. Failed: {len(failed_tasks)}/{len(task_dir_names)}")

    train_dataset, val_dataset = [], []
    for t, task_dir_name in enumerate(task_dir_names):
        if task_dir_name in failed_tasks:
            continue

        row = {}
        row["description"] = task_descriptions[t]
        row["task_dir"] = task_dir_name
        initial_test_path = Path(args.task_dir) / task_dir_name / "environment" / "test_initial_state.py"

        if t < len(task_dir_names) - args.eval_count:
            train_dataset.append(row)
        else:
            val_dataset.append(row)

    train_dataset = datasets.Dataset.from_list(train_dataset)
    val_dataset = datasets.Dataset.from_list(val_dataset)


    # 为每条样本补充训练所需的 prompt / 环境 / 奖励字段
    def make_map_fn(split):
        def process_fn(example, idx):
            system_prompt = AGENT_SYSTEM_PROMPT
            question = AGENT_USER_TEMPLATE.format(task_description=example["description"])

            data = {
                "data_source": "terminal_tasks",
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "env_class": "terminal_task",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": os.path.join(args.task_dir, example["task_dir"]),
                },
                "extra_info": {
                    "task_dir": os.path.join(args.task_dir, example["task_dir"]),
                    "max_time": args.max_time,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
