"""任务抽象：一类「整仓翻译」任务的全部差异点。

每类任务用一个 TaskSpec 描述，调度层（batch.py）只面对 TaskSpec，
不关心背后是哪种语言对、在本地还是容器里跑。
"""

import os
import posixpath
from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import config


@dataclass
class TaskSpec:
    """一类翻译任务的定义。

    属性:
        name: 任务标识（目录名的一部分）。
        source_ext: 源文件后缀，用于扫描数据集。
        target_ext: 生成物入口文件后缀。
        blocked_patterns: agent 命令黑名单正则（按任务注入）。
        block_msg: 命中黑名单时返回给 agent 的提示。
        build_prompt: 由上下文字典拼出给 agent 的完整提示词。
        find_reference: 由宿主机源文件路径定位「参考可执行文件」，
            返回 (宿主路径, 容器内建议文件名)；找不到返回 (None, None)。
    """

    name: str
    source_ext: str
    target_ext: str
    blocked_patterns: list = field(default_factory=list)
    block_msg: str = ""
    forbidden_hints: dict = field(default_factory=dict)
    build_prompt: Optional[Callable[[dict], str]] = None
    find_reference: Optional[Callable[[str], tuple]] = None


def strip_ext(rel_path, ext):
    """去掉相对路径的后缀，统一用 / 分隔。"""
    rel = rel_path.replace(os.sep, "/")
    return rel[: -len(ext)] if rel.endswith(ext) else rel


def package_dir(run_root, rel_path, source_ext):
    """某源文件对应的产物目录：output/<task>/<run>/packages/<相对路径去后缀>_pkg。"""
    return os.path.join(
        str(run_root), "packages", strip_ext(rel_path, source_ext) + "_pkg"
    )


def entry_name(rel_path, source_ext, target_ext):
    """入口文件名，如 test1.py -> test1.mjs。"""
    return os.path.basename(strip_ext(rel_path, source_ext)) + target_ext


def container_tag(rel_path, source_ext):
    """由相对路径生成容器名后缀，如 sample_lib_a/test1.py -> sample_lib_a-test1。"""
    return strip_ext(rel_path, source_ext).replace("/", "-")


def container_dataset_path(rel_path, source_ext):
    """源文件拷进容器后的展示路径（数据集根下按相对路径摆放）。"""
    return posixpath.join(config.CONTAINER_DATASET_DIR, strip_ext(rel_path, source_ext) + source_ext)


def library_of(rel_path):
    """相对路径的第一段即库名。"""
    return rel_path.replace(os.sep, "/").split("/")[0]
