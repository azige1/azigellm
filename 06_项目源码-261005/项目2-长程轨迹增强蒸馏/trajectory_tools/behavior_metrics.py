"""行为指标：盲改、工具报错率、完成宣告前未验证、压缩后首个动作、接口签名错位率。

盲改口径：Write/Edit 工具调用构成候选序列，中间出现执行验证
（Bash 命令含 pytest、python -m pytest、unittest、``python -c "import``、tox
或运行 ``test_*.py``）即切断。
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional

from trajectory_tools.token_estimate import count_text_tokens

WRITE_EDIT_TOOLS = ("write_file", "edit_file")
NAVIGATION_TOOLS = ("read_file", "glob_files", "grep_text")

VERIFICATION_PATTERNS = (
    "pytest",
    "python -m pytest",
    "unittest",
    'python -c "import',
    "tox",
)
VERIFICATION_REGEX = re.compile(r"(^|[/&'\"=\s])(test_[\w-]*\.py)([\s'\"]|$)")


def is_verification_command(command: str) -> bool:
    """判断一条 bash 命令是否为执行验证。"""
    if any(p in command for p in VERIFICATION_PATTERNS):
        return True
    return bool(VERIFICATION_REGEX.search(command))


def _bash_command_from_call(call: dict[str, Any]) -> Optional[str]:
    """从 tool_call 里取 bash 命令文本。"""
    if call.get("function", {}).get("name") != "bash":
        return None
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    return args.get("command", "")


def iter_tool_calls(messages: list[dict[str, Any]]):
    """按顺序产出 (消息下标, call) 对。"""
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            yield idx, call


def compute_blind_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    """计算 max_blind（最长连续盲改）与 blind_segments（连续 >=3 次的段数）。"""
    run = 0
    max_blind = 0
    segments = 0
    for _, call in iter_tool_calls(messages):
        name = call.get("function", {}).get("name")
        if name in WRITE_EDIT_TOOLS:
            run += 1
            max_blind = max(max_blind, run)
        elif name == "bash":
            command = _bash_command_from_call(call) or ""
            if is_verification_command(command):
                if run >= 3:
                    segments += 1
                run = 0
        # 其余工具不打断盲改
    if run >= 3:
        segments += 1
    return {"max_blind": max_blind, "blind_segments": segments}


def tool_error_rate(messages: list[dict[str, Any]]) -> dict[str, float | int]:
    """工具报错率：is_error 的工具结果数 / 工具调用总数。"""
    total = 0
    errors = 0
    for message in messages:
        if message.get("role") == "tool":
            total += 1
            if message.get("is_error"):
                errors += 1
    return {"tool_calls": total, "tool_errors": errors, "error_rate": errors / total if total else 0.0}


def completion_before_verification(
    messages: list[dict[str, Any]], claim_pattern: str = "TASK COMPLETE"
) -> bool:
    """完成宣告前是否没有执行过任何验证命令。"""
    verified = False
    for _, call in iter_tool_calls(messages):
        name = call.get("function", {}).get("name")
        if name == "bash" and is_verification_command(_bash_command_from_call(call) or ""):
            verified = True
        if name in WRITE_EDIT_TOOLS:
            pass
    if verified:
        return False
    return any(
        message.get("role") == "assistant" and claim_pattern in (message.get("content") or "")
        for message in messages
    )


def first_action_after_compact(messages: list[dict[str, Any]]) -> Optional[str]:
    """压缩后第一个工具动作：read / test / other；无压缩边界返回 None。"""
    boundary_idx = None
    for idx, message in enumerate(messages):
        if message.get("role") == "user" and "continued from a previous conversation" in (
            message.get("content") or ""
        ):
            boundary_idx = idx
    if boundary_idx is None:
        return None
    for idx, call in iter_tool_calls(messages):
        if idx <= boundary_idx:
            continue
        name = call.get("function", {}).get("name")
        if name == "read_file":
            return "read"
        if name == "bash" and is_verification_command(_bash_command_from_call(call) or ""):
            return "test"
        return "other"
    return None


def _public_signatures(source: str) -> dict[str, str]:
    """提取公开函数与方法签名字符串（参数名、顺序、默认值、*args/**kwargs）。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}

    def fmt(node):
        return ast.unparse(node.args)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            out[f"def {node.name}"] = fmt(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"class {node.name}.{sub.name}"] = fmt(sub)
    return out


def signature_mismatch_rate(
    base_repo: Path, final_repo: Path, src_dir: str, base_commit: str
) -> dict[str, int | float]:
    """骨架公开签名在最终工作区中错位（不一致或被删除）的比例。"""
    import git

    repo = git.Repo(base_repo)
    base_tree = repo.commit(base_commit).tree
    total = 0
    mismatch = 0
    for entry in base_tree[src_dir].traverse():
        if not str(entry.path).endswith(".py"):
            continue
        try:
            base_source = (base_tree / str(entry.path)).data_stream.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        base_sigs = _public_signatures(base_source)
        final_file = final_repo / str(entry.path)
        final_sigs = (
            _public_signatures(final_file.read_text(encoding="utf-8", errors="replace"))
            if final_file.exists()
            else {}
        )
        for name, sig in base_sigs.items():
            total += 1
            if final_sigs.get(name) != sig:
                mismatch += 1
    return {"total_signatures": total, "mismatched": mismatch, "mismatch_rate": mismatch / total if total else 0.0}


def single_tool_result_tokens(messages: list[dict[str, Any]]) -> list[int]:
    """每条工具结果的 token 估算列表（注入触发点 token_spike 用）。"""
    return [
        count_text_tokens(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    ]


__all__ = []
