#!/usr/bin/env python3
"""判分入口：对 agent 产物做黑盒行为判分。

用法示例:
    # py2node，判分容器内执行
    python grade.py --task py2node --mode docker -m your-model

    # cpp2rust，宿主机直接执行
    python grade.py --task cpp2rust --mode local -m your-model

判分结果（逐文件通过率）写入 grading 结果目录，可用 stats_report.py 再汇总。
"""

import argparse

from harness import config
from harness.tasks import get_task_spec, task_names
from grading.runners import ContainerRunner, LocalRunner
from grading.score import run_cpp2rust_grading, run_py2node_grading


def make_runner(mode, image, timeout):
    """按模式构造执行后端。"""
    if mode == "docker":
        return ContainerRunner(image, timeout=timeout)
    return LocalRunner()


def main():
    parser = argparse.ArgumentParser(description="跨语言仓库重建：行为判分")
    parser.add_argument("--task", required=True, choices=task_names(), help="任务类型")
    parser.add_argument("--mode", choices=["local", "docker"], default="docker")
    parser.add_argument("-m", "--model", default=None, help="被判分的模型/运行名")
    parser.add_argument("--dataset-root", default=None, help="数据集目录（默认 data/dataset/<task>）")
    parser.add_argument("--output-root", default=None, help="产物目录（默认 output/<task>/<model>）")
    parser.add_argument("--cases-dir", default=None, help="用例目录（默认 data/testcases/<task>）")
    parser.add_argument("--results-dir", default=None, help="结果目录（默认 output/<task>/<model>/grade_results）")
    parser.add_argument("--image", default=None, help="判分镜像（默认读 REBUILD_DOCKER_IMAGE）")
    parser.add_argument("--timeout", type=int, default=5, help="单次执行超时秒数")
    parser.add_argument("--api-lines", default=None,
                        help="cpp2rust 难度分桶用的 api_lines.jsonl（默认用例目录下的 api_lines.jsonl）")
    args = parser.parse_args()

    spec = get_task_spec(args.task)
    model_name = args.model or config.LLM_MODEL
    dataset_root = args.dataset_root or str(config.DATASET_DIR / spec.name)
    output_root = args.output_root or str(config.OUTPUT_DIR / spec.name / model_name)
    cases_dir = args.cases_dir or str(config.CASES_DIR / spec.name)
    results_dir = args.results_dir or str(config.OUTPUT_DIR / spec.name / model_name / "grade_results")
    image = args.image or config.docker_image(spec.name)

    print(f"[CONFIG] 任务: {spec.name}  模式: {args.mode}  模型: {model_name}")
    print(f"[CONFIG] 数据集: {dataset_root}")
    print(f"[CONFIG] 产物目录: {output_root}")
    print(f"[CONFIG] 用例目录: {cases_dir}")
    if args.mode == "docker":
        print(f"[CONFIG] 判分镜像: {image}")
    print()

    runner = make_runner(args.mode, image, args.timeout)
    try:
        if spec.name == "py2node":
            run_py2node_grading(cases_dir, dataset_root, output_root, results_dir, runner,
                                timeout=args.timeout)
        else:
            import os
            case_files = sorted(
                os.path.join(cases_dir, f) for f in os.listdir(cases_dir)
                if f.endswith(".jsonl") and f != "api_lines.jsonl"
            )
            if not case_files:
                print(f"[ERROR] 用例目录里没有可用的 jsonl: {cases_dir}")
                return
            api_lines = args.api_lines or os.path.join(cases_dir, "api_lines.jsonl")
            cache_path = os.path.join(output_root, "grade_cache.json")
            for cases_path in case_files:
                print(f"\n##### 用例文件: {os.path.basename(cases_path)} #####")
                run_cpp2rust_grading(
                    cases_path, dataset_root, os.path.join(output_root, "packages"),
                    cache_path, runner, api_lines_path=api_lines, timeout=args.timeout,
                )
    finally:
        if isinstance(runner, ContainerRunner):
            runner.close()


if __name__ == "__main__":
    main()
