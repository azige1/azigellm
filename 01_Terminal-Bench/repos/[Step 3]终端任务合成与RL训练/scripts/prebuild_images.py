"""预构建训练数据的 Docker 镜像。

读取 rl/prepare_data.py 产出的 train/validation parquet，对其中的任务目录
并行执行 docker build，已存在的镜像自动跳过。

用法：
    python scripts/prebuild_images.py --data-dir ./data [--workers 8]
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def build_image(task_dir):
    dockerfile = f'{task_dir}/environment/Dockerfile'
    if not os.path.exists(dockerfile):
        return task_dir, 'no_dockerfile'
    tag = f'rl-prebuild-{os.path.basename(task_dir)}'
    r = subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True)
    if r.returncode == 0:
        return task_dir, 'skipped'
    r = subprocess.run(
        ['docker', 'build', '-t', tag, '-f', dockerfile, f'{task_dir}/environment', '--quiet'],
        capture_output=True, text=True, timeout=300
    )
    return task_dir, 'built' if r.returncode == 0 else f'failed: {r.stderr[-200:]}'


def main():
    parser = argparse.ArgumentParser(description="Pre-build Docker images for training tasks")
    parser.add_argument('--data-dir', default='./data', help='Directory containing train.parquet / validation.parquet')
    parser.add_argument('--workers', type=int, default=8, help='Parallel build workers')
    args = parser.parse_args()

    print("Loading parquet...")
    train = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    val = pd.read_parquet(os.path.join(args.data_dir, 'validation.parquet'))
    task_dirs = list(set(
        list(train['extra_info'].apply(lambda x: x['task_dir'])) +
        list(val['extra_info'].apply(lambda x: x['task_dir']))
    ))
    print(f"Pre-building {len(task_dirs)} Docker images with {args.workers} parallel workers...")

    counts = {'built': 0, 'skipped': 0, 'failed': 0, 'no_dockerfile': 0}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_image, td): td for td in task_dirs}
        for future in as_completed(futures):
            task_dir, status = future.result()
            done += 1
            key = status.split(':')[0]
            counts[key] = counts.get(key, 0) + 1
            if status == 'built' or status.startswith('failed'):
                print(f"[{done}/{len(task_dirs)}] {status}: {os.path.basename(task_dir)}")

    print(f"\nDone. {counts}")


if __name__ == '__main__':
    main()
