"""列出某个仓库的测试用例 ID（离线数据，随包分发）。

数据存放在 ``rebuild_spec/data/test_ids/`` 下的 bz2 压缩文本里：
- 整库重建仓库：``<repo>.bz2``，每行一个 pytest 用例 ID；
- swe 风格实例：``<instance>#fail_to_pass.bz2`` 与 ``<instance>#pass_to_pass.bz2`` 两个文件。
"""

import bz2
from typing import List
import os

import rebuild_spec


def _read_bz2(bz2_file: str) -> str:
    with bz2.open(bz2_file, "rt") as f:
        out = f.read()
    return out


def main(repo: str, verbose: int) -> List[List[str]]:
    repo = repo.lower()
    repo = repo.replace(".", "-")
    pkg_dir = os.path.dirname(rebuild_spec.__file__)
    if "__" in repo:
        in_file_fail = _read_bz2(f"{pkg_dir}/data/test_ids/{repo}#fail_to_pass.bz2")
        in_file_pass = _read_bz2(f"{pkg_dir}/data/test_ids/{repo}#pass_to_pass.bz2")
    else:
        in_file_fail = _read_bz2(f"{pkg_dir}/data/test_ids/{repo}.bz2")
        in_file_pass = ""
    out = [in_file_fail, in_file_pass]
    if verbose:
        print(f"FAIL TO PASS:\n{out[0]}\nPASS TO PASS:\n{out[1]}")
    out = [out[0].split("\n"), out[1].split("\n")]
    return out


__all__ = []
