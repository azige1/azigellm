"""agent_rollout 的七个工具：Bash / Read / Write / Edit / Glob / Grep / TodoWrite。

全部以 OpenAI function calling 格式声明；工具出错时返回带
``is_error: true`` 的结果。Bash 在任务虚拟环境下执行，且不允许访问隐藏测试目录。
"""

import fnmatch
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MAX_BASH_OUTPUT_CHARS = 10_000
BASH_TIMEOUT_SECONDS = 120


@dataclass
class ToolResult:
    """一次工具执行的结果。"""

    content: str
    is_error: bool = False


@dataclass
class Workspace:
    """工具执行上下文：工作区根目录、虚拟环境 bin 目录与禁入路径。"""

    root: Path
    venv_bin: Optional[Path] = None
    forbidden_dirs: list[Path] = field(default_factory=list)
    todo_items: list[str] = field(default_factory=list)

    def bash_env(self) -> dict:
        """构造 Bash 环境：把任务虚拟环境 bin 前置到 PATH。"""
        env = dict(os.environ)
        if self.venv_bin is not None:
            env["PATH"] = f"{self.venv_bin}:{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(self.venv_bin.parent)
        return env


def _check_path(workspace: Workspace, path: str) -> Path:
    target = (workspace.root / path).resolve()
    if not str(target).startswith(str(workspace.root.resolve())):
        raise PermissionError(f"路径越出工作区：{path}")
    for forbidden in workspace.forbidden_dirs:
        if str(target).startswith(str(Path(forbidden).resolve())):
            raise PermissionError(f"不允许访问路径：{path}")
    return target


def run_bash(workspace: Workspace, command: str) -> ToolResult:
    """在工作区内执行 bash 命令，超时 120 秒，输出截断到 1 万字符。"""
    # 命令文本中直接出现禁入路径（任务目录、隐藏测试目录）时拒绝执行
    for forbidden in workspace.forbidden_dirs:
        forbidden_text = str(Path(forbidden).resolve())
        if forbidden_text in command or "hidden_tests" in command:
            return ToolResult("不允许访问该路径", is_error=True)
    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=str(workspace.root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=workspace.bash_env(),
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=BASH_TIMEOUT_SECONDS)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = proc.communicate()
        return ToolResult(
            f"命令超时（{BASH_TIMEOUT_SECONDS} 秒）已被终止\n" + (output or "")[:MAX_BASH_OUTPUT_CHARS],
            is_error=True,
        )
    if output is None:
        output = ""
    if len(output) > MAX_BASH_OUTPUT_CHARS:
        output = output[:MAX_BASH_OUTPUT_CHARS] + "\n[输出已截断]"
    text = output + f"\n[退出码 {exit_code}]"
    return ToolResult(text, is_error=exit_code != 0)


def read_file(workspace: Workspace, path: str, offset: int = 0, limit: Optional[int] = None) -> ToolResult:
    """读文件，返回带行号的内容；offset 为起始行（从 0 计）。"""
    target = _check_path(workspace, path)
    if not target.is_file():
        return ToolResult(f"文件不存在：{path}", is_error=True)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    end = None if limit is None else offset + limit
    numbered = [f"{i + 1}\t{line}" for i, line in enumerate(lines[offset:end])]
    return ToolResult("\n".join(numbered))


def write_file(workspace: Workspace, path: str, content: str) -> ToolResult:
    """写文件（覆盖），父目录自动创建。"""
    target = _check_path(workspace, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(f"已写入 {path}（{len(content)} 字符）")


def edit_file(workspace: Workspace, path: str, old_string: str, new_string: str) -> ToolResult:
    """精确替换：old_string 必须唯一匹配，否则返回错误。"""
    target = _check_path(workspace, path)
    if not target.is_file():
        return ToolResult(f"文件不存在：{path}", is_error=True)
    content = target.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_string)
    if count == 0:
        return ToolResult("old_string 未匹配到任何内容", is_error=True)
    if count > 1:
        return ToolResult(f"old_string 匹配到 {count} 处，不唯一", is_error=True)
    target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return ToolResult(f"已替换 {path} 中的 1 处内容")


def glob_files(workspace: Workspace, pattern: str) -> ToolResult:
    """按通配模式列出工作区内文件。"""
    matches = sorted(
        str(p.relative_to(workspace.root))
        for p in workspace.root.rglob("*")
        if p.is_file()
        and fnmatch.fnmatch(str(p.relative_to(workspace.root)), pattern)
        and ".git" not in p.parts
    )
    return ToolResult("\n".join(matches[:500]) or "(无匹配)")


def grep_text(workspace: Workspace, pattern: str, path: str = ".") -> ToolResult:
    """在工作区内按正则搜索文本，返回 path:line: content。"""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return ToolResult(f"正则表达式无效：{e}", is_error=True)
    base = _check_path(workspace, path)
    files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts]
    hits = []
    for f in files:
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines()):
                if regex.search(line):
                    hits.append(f"{f.relative_to(workspace.root)}:{i + 1}: {line.strip()}")
                    if len(hits) >= 500:
                        return ToolResult("\n".join(hits) + "\n[结果已截断]")
        except OSError:
            continue
    return ToolResult("\n".join(hits) or "(无匹配)")


def todo_write(workspace: Workspace, items: list[str]) -> ToolResult:
    """维护规划清单（整表覆盖），供过程监督的 plan_update / phase_boundary 触发点使用。"""
    workspace.todo_items = list(items)
    return ToolResult("已更新规划清单（{} 项）".format(len(items)))


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "在工作区内执行 bash 命令（任务虚拟环境下，超时 120 秒）",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "要执行的命令"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读文件（带行号），支持 offset/limit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写文件（覆盖），父目录自动创建",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "精确字符串替换：old_string 必须唯一匹配",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "按通配模式列出工作区内文件",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_text",
            "description": "按正则搜索文本，返回 path:line: content",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "维护规划清单（整表覆盖）",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["items"],
            },
        },
    },
]

_HANDLERS = {
    "bash": run_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "glob_files": glob_files,
    "grep_text": grep_text,
    "todo_write": todo_write,
}


def execute_tool(workspace: Workspace, name: str, arguments_json: str) -> ToolResult:
    """按名字分发执行工具调用，任何异常都转为 is_error 结果。"""
    handler = _HANDLERS.get(name)
    if handler is None:
        return ToolResult(f"未知工具：{name}", is_error=True)
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return ToolResult(f"参数不是合法 JSON：{e}", is_error=True)
    try:
        return handler(workspace, **arguments)
    except PermissionError as e:
        return ToolResult(str(e), is_error=True)
    except TypeError as e:
        return ToolResult(f"参数错误：{e}", is_error=True)
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"工具执行异常：{e}", is_error=True)


__all__ = []
