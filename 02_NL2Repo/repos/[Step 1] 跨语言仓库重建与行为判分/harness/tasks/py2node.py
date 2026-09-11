"""py2node 任务：把 Python 库逐文件重写成 Node.js ESM（.mjs）。

约束要点（写进给 agent 的提示词）：
- 只允许 Node.js 内置模块，禁止任何 npm 包；
- CLI 参数与原 Python 文件完全对齐；
- stdout 与原实现逐字节一致。
"""

import os

from .. import config, sandbox
from .base import TaskSpec

# 每个库「禁止参照的现成实现」提示：样例数据只保留一个库
FORBIDDEN_HINTS = {
    "sample_lib_a": "Uint8Array / Buffer 上的现成编解码库",
}

_PROMPT_TEMPLATE = """
You are a senior cross-language migration engineer (Python to Node.js). Read the following Python source code:

--- Source ({source_path}) ---
{source_code}
---

Requirements:
1. **Runtime & module style**: Write plain JavaScript for Node.js using ES Modules only (`import` / `export`). Never use `require()` or `module.exports`. Every file you generate (library modules and the entry file) must carry the `.mjs` suffix.

2. **CLI compatibility**: The entry file must accept exactly the same command-line arguments as the Python file — same option names, defaults and required flags. Parse `process.argv` by hand so that `node {entry_name} --arg val` behaves exactly like the reference program.

3. **Behavioural equivalence**: Algorithm logic, numeric precision and string formatting must match the Python original. `console.log` output must be character-for-character identical to Python's `print` output (including spaces and newlines).

4. **No external packages**: Any npm dependency (yargs, argparse, ...) is forbidden, and `import` may only reference local files. Stick to Node.js built-ins such as `node:fs`, `node:path`, `node:url`. Do not embed or shell out to Python.

5. **Code layout**: Place all generated files under `{out_dir}`. Split the library into several focused modules that expose their API via `export`; remember that ESM imports need the full file extension (e.g. `import {{ x }} from './utils.mjs'`). Your shell working directory is `{work_dir}`, but all deliverables go to `{out_dir}`.

6. **Black-box rule**: Do not look up the internals of the original Python package, and do not call or mention {forbidden_hint}. Re-implement the behaviour from scratch.

{reference_section}
Finish by writing the entry file `{entry_name}` into `{out_dir}`.
"""

_REFERENCE_SECTION = """7. **Probing the reference**: A pre-compiled build of the original program is available at `{reference_path}`. Run it with any arguments to observe the expected behaviour. Invoke it directly — do not use the `python` command."""


def build_prompt(ctx):
    """按运行方式（本地/容器）拼提示词。"""
    reference_section = ""
    if ctx.get("reference_path"):
        reference_section = _REFERENCE_SECTION.format(reference_path=ctx["reference_path"])
    return _PROMPT_TEMPLATE.format(
        source_path=ctx["source_path"],
        source_code=ctx["source_code"],
        entry_name=ctx["entry_name"],
        out_dir=ctx["out_dir"],
        work_dir=ctx["work_dir"],
        forbidden_hint=ctx.get("forbidden_hint", "any existing third-party implementation"),
        reference_section=reference_section,
    )


def find_reference(source_path):
    """参考可执行文件与源文件同目录，命名为 <模块名>_executable。"""
    module = os.path.splitext(os.path.basename(source_path))[0]
    candidate = os.path.join(os.path.dirname(source_path), f"{module}_executable")
    if os.path.exists(candidate):
        return candidate, f"{module}_executable"
    return None, None


SPEC = TaskSpec(
    name="py2node",
    source_ext=".py",
    target_ext=".mjs",
    blocked_patterns=sandbox.NPM_IMPORT_PATTERNS,
    block_msg=sandbox.NPM_BLOCK_MSG,
    forbidden_hints=FORBIDDEN_HINTS,
    build_prompt=build_prompt,
    find_reference=find_reference,
)
