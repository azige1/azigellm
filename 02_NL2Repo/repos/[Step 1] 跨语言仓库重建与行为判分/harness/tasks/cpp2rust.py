"""cpp2rust 任务：把 C++ 库的测试程序逐文件重写成纯 std 的 Rust。

约束要点（写进给 agent 的提示词）：
- 只用 Rust 标准库，禁止 crates.io 依赖；
- CLI 参数与原 C++ 二进制完全对齐；
- println! 输出与 std::cout 逐字节一致；
- 产物需在产物目录里编译出可执行文件。
"""

import os
import re

from .. import config, sandbox
from .base import TaskSpec

# 每个库「禁止使用的现成 crate」提示；样例数据只保留一个库，无特殊禁用项
FORBIDDEN_HINTS = {
    "sample_lib_a": "none",
}

_PROMPT_TEMPLATE = """
You are a senior systems migration engineer (C++ to Rust). Read the following C++ source code:

--- Source ({source_path}) ---
{source_code}
---

Requirements:
1. **Toolchain**: Write pure Rust (2021 edition). The code must build with `rustc` directly or as a Cargo project.

2. **CLI compatibility**: The Rust binary must accept exactly the same command-line arguments as the C++ binary — same option names, defaults and required flags. Parse arguments via `std::env::args()`.

3. **Behavioural equivalence**: Algorithm logic, numeric precision and string formatting must match the C++ original. `println!` output must be character-for-character identical to the original `std::cout` output.

4. **No external crates**: Only the standard library (`std`) is allowed — nothing from crates.io. Forbidden crates for this repository: {forbidden_hint}. Implement everything from scratch.

5. **Code layout**: Create a complete Cargo project under `{out_dir}`, organise the library into modules, and place the entry file `{entry_name}` in `{out_dir}`.

6. **Black-box rule**: Do not look up the internals of the original C++ library. Infer the behaviour from its interface and re-implement with `std`.

7. **Deliverables**:
   - Implement all library modules inside `{out_dir}`.
   - Save the entry file as `{out_dir}/{entry_name}`.
   - Build it so that an executable exists at `{out_dir}/{entry_name}` (via `rustc {entry_name}` or `cargo build --release`).

{reference_section}
"""

_REFERENCE_SECTION = """Hint: `{reference_path}` is the compiled C++ binary — run it with any arguments to probe the expected behaviour while debugging."""


def build_prompt(ctx):
    reference_section = ""
    if ctx.get("reference_path"):
        reference_section = _REFERENCE_SECTION.format(reference_path=ctx["reference_path"])
    return _PROMPT_TEMPLATE.format(
        source_path=ctx["source_path"],
        source_code=ctx["source_code"],
        entry_name=ctx["entry_name"],
        out_dir=ctx["out_dir"],
        forbidden_hint=ctx.get("forbidden_hint", "none"),
        reference_section=reference_section,
    )


def find_reference(source_path):
    """按多个候选位置找参考可执行文件。

    数据集中编译产物可能在源文件同目录或上一级 tests/ 下，
    命名可能是 <模块名>_executable 或直接 <模块名>。
    """
    module = os.path.splitext(os.path.basename(source_path))[0]
    src_dir = os.path.dirname(source_path)
    parent_dir = os.path.dirname(src_dir)
    candidates = [
        os.path.join(src_dir, f"{module}_executable"),
        os.path.join(src_dir, module),
        os.path.join(parent_dir, "tests", f"{module}_executable"),
        os.path.join(parent_dir, "tests", module),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, os.path.basename(path)
    return None, None


def test_number(rel_path):
    """从 testN.cpp 形式的文件名里取 N，用于产物命名；不匹配返回 None。"""
    match = re.search(r"test(\d+)\.cpp", os.path.basename(rel_path))
    return match.group(1) if match else None


SPEC = TaskSpec(
    name="cpp2rust",
    source_ext=".cpp",
    target_ext=".rs",
    blocked_patterns=sandbox.CARGO_DEP_PATTERNS,
    block_msg=sandbox.CARGO_BLOCK_MSG,
    forbidden_hints=FORBIDDEN_HINTS,
    build_prompt=build_prompt,
    find_reference=find_reference,
)
