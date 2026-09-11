#!/usr/bin/env python3
"""做题入口：让 agent 批量完成整仓翻译任务。

用法示例:
    # py2node 任务，Docker 断网容器，4 并发
    python run_benchmark.py --task py2node --mode docker -k 4 -m your-model

    # cpp2rust 任务，本地执行
    python run_benchmark.py --task cpp2rust --mode local -m your-model

    # py2node 本地自修复模式：每题最多重做 3 次
    python run_benchmark.py --task py2node --mode local --self-repair 3 -m your-model

开跑前会打印第一题的完整提示词，回车确认、输入 q 退出（--yes 跳过确认）。
产物写入 output/<task>/<model>/，轨迹写入 logs/<task>/。
"""

import argparse
import os
import sys

from harness import config
from harness.batch import collect_jobs, run_batch, sample_prompt
from harness.tasks import get_task_spec, task_names


def main():
    parser = argparse.ArgumentParser(description="跨语言仓库重建：批量做题 runner")
    parser.add_argument("--task", required=True, choices=task_names(), help="任务类型")
    parser.add_argument("--mode", choices=["local", "docker"], default="docker",
                        help="本地直接跑，或在断网 Docker 容器里跑（默认 docker）")
    parser.add_argument("-m", "--model", default=None,
                        help="模型名（默认取环境变量 LLM_MODEL）")
    parser.add_argument("-k", "--workers", type=int, default=4, help="并发进程数（默认 4）")
    parser.add_argument("--self-repair", type=int, default=0, metavar="N",
                        help="本地自修复次数上限（仅 py2node 本地模式，0 表示关闭）")
    parser.add_argument("--dataset-root", default=None, help="数据集目录（默认 data/dataset/<task>）")
    parser.add_argument("--dry-run", action="store_true", help="只列出待做题，不真正执行")
    parser.add_argument("--yes", action="store_true", help="跳过提示词确认")
    args = parser.parse_args()

    spec = get_task_spec(args.task)
    model_name = args.model or config.LLM_MODEL

    if args.self_repair and (args.task != "py2node" or args.mode != "local"):
        parser.error("--self-repair 目前只支持 py2node 的本地模式")

    dataset_root = args.dataset_root or str(config.DATASET_DIR / spec.name)
    run_root = config.OUTPUT_DIR / spec.name / model_name
    os.makedirs(run_root / "packages", exist_ok=True)

    print(f"[CONFIG] 任务: {spec.name}  模式: {args.mode}")
    print(f"[CONFIG] 模型: {model_name}  并发: {args.workers}")
    if args.mode == "docker":
        print(f"[CONFIG] 判分镜像: {config.docker_image(spec.name)}")
    print(f"[CONFIG] 数据集: {dataset_root}")
    print(f"[CONFIG] 产物目录: {run_root}")

    jobs = collect_jobs(spec.name, args.mode, dataset_root, run_root, model_name,
                        self_repair=args.self_repair)
    if not jobs:
        print("没有待处理的题目。")
        return
    print(f"共 {len(jobs)} 题待处理。")

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(f"  {i}. {job['rel_path']}")
        return

    if not args.yes:
        print("=" * 60)
        print("第一题的示例提示词：")
        print("=" * 60)
        print(sample_prompt(jobs[0]))
        print("=" * 60)
        answer = input("\n回车开始，输入 q 退出: ")
        if answer.lower() == "q":
            print("已退出。")
            return

    run_batch(jobs, args.workers, run_root)


if __name__ == "__main__":
    main()
