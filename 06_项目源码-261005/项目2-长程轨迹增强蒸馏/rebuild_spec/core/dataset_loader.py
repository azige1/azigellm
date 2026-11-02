"""数据集加载：支持 Hugging Face 数据集名与本地任务目录。

本地任务目录约定：``{dataset_name}/{split}.jsonl`` 存在时按 json 构建器
读取（无需数据集脚本），否则原样走 ``load_dataset``。
"""

from pathlib import Path
from typing import Iterator, Union

from datasets import load_dataset

from rebuild_spec.core.constants import RepoRecord, FunctionRecord


def load_task_records(
    dataset_name: str, dataset_split: str
) -> Iterator[Union[RepoRecord, FunctionRecord]]:
    """按名字或本地目录加载数据集记录。"""
    split_file = Path(dataset_name) / f"{dataset_split}.jsonl"
    if Path(dataset_name).is_dir() and split_file.exists():
        return load_dataset(
            "json", data_files={dataset_split: str(split_file)}, split=dataset_split
        )  # type: ignore
    return load_dataset(dataset_name, split=dataset_split)  # type: ignore


__all__ = []
