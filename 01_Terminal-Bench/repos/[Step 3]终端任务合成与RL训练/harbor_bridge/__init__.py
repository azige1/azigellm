"""Harbor 基准框架的 agent 桥接层。

``shell_agent.TerminalShellAgent`` 把「单命令终端 agent」协议接入 Harbor
的 BaseAgent 接口，使本地 vLLM 部署的模型可以直接在 Harbor 任务上评测。
"""
from __future__ import annotations

from typing import Any

__all__ = ["TerminalShellAgent"]


def __getattr__(name: str) -> Any:
    # 惰性导出：shell_agent 依赖 transformers/harbor 等重依赖，
    # 只在真正用到 agent 时才导入
    if name == "TerminalShellAgent":
        from .shell_agent import TerminalShellAgent

        return TerminalShellAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
