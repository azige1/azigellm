"""第二步：为选定分组构建全部 Docker 镜像。"""

import logging

import docker
from datasets import load_dataset
from typing import Iterator, Union

from rebuild_spec.core.constants import RepoRecord, FunctionRecord, SPLIT
from rebuild_spec.core.images import build_all_repo_images
from rebuild_spec.core.specs import build_spec

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def classify_dataset(dataset_name: str) -> str:
    """根据数据集名字判断记录类型：swebench / simple / scratch。"""
    name = dataset_name.lower()
    if "swe" in name:
        return "swebench"
    if (
        "humaneval" in name
        or "mbpp" in name
        or "bigcodebench" in name
        or "codecontests" in name
    ):
        return "simple"
    return "scratch"


def main(
    dataset_name: str,
    dataset_split: str,
    split: str,
    num_workers: int,
    verbose: int,
) -> None:
    dataset: Iterator[Union[RepoRecord, FunctionRecord]] = load_dataset(
        dataset_name, split=dataset_split
    )  # type: ignore
    specs = []
    dataset_name = dataset_name.lower()
    dataset_type = classify_dataset(dataset_name)
    for example in dataset:
        if "swe" in dataset_name or dataset_type == "simple":
            if split != "all" and split not in example["instance_id"]:
                continue
        else:
            repo_name = example["repo"].split("/")[-1]
            if split != "all" and repo_name not in SPLIT[split]:
                continue
        spec = build_spec(example, dataset_type, absolute=True)
        specs.append(spec)

    client = docker.from_env()
    build_all_repo_images(client, specs, dataset_type, num_workers, verbose)


__all__ = []
