"""工具返回报错规范化：提取首个失败用例的「异常类型 + 函数名 + 断言形态」作为 GiGPO anchor。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# 匹配 pytest 失败行或 Python traceback 末行的异常信息
# 异常名：带前缀的 *Error / *Exception / *Warning / *Exit / *Interrupt，或恰为 Exception、StopIteration
_EXCEPTION_NAME = (
    r"(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit|Interrupt)"
    r"|(?:[A-Za-z_][A-Za-z0-9_.]*\.)?(?:Exception|StopIteration))"
)
_EXCEPTION_LINE_RE = re.compile(rf"^(?:E\s+)?({_EXCEPTION_NAME})\s*:\s*(.*)$")
_BARE_EXCEPTION_RE = re.compile(rf"^(?:E\s+)?({_EXCEPTION_NAME})\s*$")
# 匹配 traceback File "...", line N, in func_name
_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+\d+,\s+in\s+([A-Za-z_][A-Za-z0-9_<>]*)')
# 匹配 pytest 长格式栈帧：src/pkg/core.py:12: in parse
_PYTEST_FRAME_RE = re.compile(r"^([^\s:]+\.py):\d+:\s+in\s+([A-Za-z_][A-Za-z0-9_<>]*)\s*$")
# 匹配 pytest 短摘要行：FAILED tests/test_core.py::test_parse - KeyError: 'sep'
_PYTEST_FAILED_SUMMARY_RE = re.compile(
    r"^FAILED\s+([^\s:]+)::([^\s-]+)\s*-\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(?::\s*(.*))?$"
)
# 匹配 pytest 详情节标题：_____________________ test_name _____________________
_PYTEST_SECTION_RE = re.compile(r"^_{3,}\s+([^\s_][^\n]*?)\s+_{3,}$")
# 匹配断言行
_ASSERT_STMT_RE = re.compile(r"^\s*(?:E\s+|>\s+)?assert\s+(.+)$")
_PYTEST_RAISES_RE = re.compile(r"pytest\.raises\(\s*([A-Za-z_][A-Za-z0-9_.]*)")

# 字面量与易变片段规范化正则
_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
_TMP_DIR_RE = re.compile(r"/(?:tmp|var/tmp)/[A-Za-z0-9_.-]+")
_ABS_REPO_PREFIX_RE = re.compile(r"(?:/testbed|/workspace|/root/[^\s/]+|/home/[^\s/]+)/")
_LINE_NO_RE = re.compile(r"\bline\s+\d+|:\d+(?=:|\b)")
_STRING_LIT_RE = re.compile(r"""(?:'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")""")
_NUMBER_LIT_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_])")


def normalize_error_text(text: str) -> str:
    """将路径、行号、地址、临时目录与字面量替换为占位符。"""
    cleaned = _TMP_DIR_RE.sub("<tmp>", text)
    cleaned = _ABS_REPO_PREFIX_RE.sub("", cleaned)
    cleaned = _HEX_ADDR_RE.sub("<addr>", cleaned)
    cleaned = _STRING_LIT_RE.sub("<str>", cleaned)
    cleaned = _LINE_NO_RE.sub(lambda m: "line <line>" if m.group(0).startswith("line") else ":<line>", cleaned)
    cleaned = _NUMBER_LIT_RE.sub("<num>", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_assertion_shape(assert_expr: str) -> str:
    """把具体 assert 表达式规整为结构化断言形态。"""
    expr = assert_expr.strip()
    for op in ("==", "!=", ">=", "<=", " not in ", " in ", " is not ", " is ", ">", "<"):
        if op in expr:
            return f"assert <expr> {op.strip()} <expr>"
    return "assert <expr>"


def _extract_first_failure_block(output: str) -> str:
    """存在多个失败用例时只截出第一个失败区块，避免用例顺序影响 anchor。"""
    lines = output.splitlines()
    fail_starts: list[int] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if _PYTEST_SECTION_RE.match(stripped) or stripped.startswith("FAILED "):
            fail_starts.append(idx)
    if len(fail_starts) >= 2 and _PYTEST_SECTION_RE.match(lines[fail_starts[0]].strip()):
        return "\n".join(lines[fail_starts[0] : fail_starts[1]])
    return output


def extract_error_anchor(
    tool_output: str,
    *,
    granularity: Literal["full", "exception_only"] = "full",
) -> str | None:
    """从工具输出提取规范化报错 anchor。

    未检测到 Python/pytest 异常时返回 None。
    - ``full``：``异常类型 + 断言形态 + 出错函数``（如 ``KeyError: <str> in parse()``）。
    - ``exception_only``：仅保留异常类型（用于规范化粒度消融）。
    """
    if not tool_output or not tool_output.strip():
        return None

    block = _extract_first_failure_block(tool_output)
    lines = [line.strip() for line in block.splitlines() if line.strip()]

    exc_type: str | None = None
    exc_detail: str = ""
    func_name: str | None = None
    assert_shape: str | None = None

    # 1. 优先检查 FAILED 摘要行（单行 pytest 输出常见形式）
    for line in lines:
        m_summary = _PYTEST_FAILED_SUMMARY_RE.match(line)
        if m_summary:
            test_func = m_summary.group(2).split("[", 1)[0]
            exc_type = m_summary.group(3).rsplit(".", 1)[-1]
            exc_detail = m_summary.group(4) or ""
            func_name = func_name or test_func
            break

    # 2. 扫描 traceback 栈帧与断言/异常行
    frame_funcs: list[str] = []
    for line in lines:
        m_frame = _TRACEBACK_FRAME_RE.search(line) or _PYTEST_FRAME_RE.match(line)
        if m_frame:
            file_path, fname = m_frame.group(1), m_frame.group(2)
            is_framework = "pluggy" in file_path or "_pytest" in file_path
            is_test_file = Path(file_path).name.startswith("test_") or "/tests/" in f"/{file_path}"
            if fname != "<module>" and not is_framework and not is_test_file:
                frame_funcs.append(fname)

        m_raise = _PYTEST_RAISES_RE.search(line)
        if m_raise and assert_shape is None:
            assert_shape = f"pytest.raises({m_raise.group(1).rsplit('.', 1)[-1]})"

        m_assert = _ASSERT_STMT_RE.match(line)
        if m_assert and assert_shape is None:
            assert_shape = _normalize_assertion_shape(m_assert.group(1))

        m_exc = _EXCEPTION_LINE_RE.match(line)
        if m_exc:
            exc_type = m_exc.group(1).rsplit(".", 1)[-1]
            exc_detail = m_exc.group(2).strip()
        else:
            m_bare = _BARE_EXCEPTION_RE.match(line)
            if m_bare and exc_type is None:
                exc_type = m_bare.group(1).rsplit(".", 1)[-1]

    if exc_type is None:
        return None

    if granularity == "exception_only":
        return exc_type

    # 取非测试框架的最深层业务函数名；若全是测试函数则取最后一个栈帧函数
    if frame_funcs:
        non_test = [f for f in frame_funcs if not f.startswith("test_")]
        func_name = non_test[-1] if non_test else frame_funcs[-1]
    elif func_name is None:
        for line in lines:
            m_sec = _PYTEST_SECTION_RE.match(line)
            if m_sec:
                func_name = m_sec.group(1).split(".")[-1].split("[")[0]
                break

    location = f" in {func_name}()" if func_name else ""
    if assert_shape is not None and exc_type == "AssertionError":
        return f"{exc_type}: {assert_shape}{location}"
    norm_detail = normalize_error_text(exc_detail) if exc_detail else ""
    if len(norm_detail) > 64:
        norm_detail = norm_detail[:64].rstrip()
    if norm_detail:
        return f"{exc_type}: {norm_detail}{location}"
    if assert_shape is not None:
        return f"{exc_type}: {assert_shape}{location}"
    return f"{exc_type}{location}"
