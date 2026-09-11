"""自定义 Harbor agent：Terminus-2，但 LLM 后端替换为网关 Bedrock 接口。

用法（harbor CLI）：
    harbor run \
        --path harbor_tasks \
        --agent-import-path task_factory.llm.bedrock_agent:BedrockTerminus2 \
        --model claude_opus \
        --n-attempts 2 \
        --n-concurrent 2 \
        --jobs-dir harbor_jobs \
        --yes
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2

from task_factory.llm.bedrock_llm import BedrockClaudeLLM


class BedrockTerminus2(Terminus2):
    """把 LiteLLM 后端换成网关 Bedrock 后端的 Terminus-2。"""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        gateway_model = model_name or "claude_opus"

        _LITELLM_NAMES = {
            "claude_4_5": "anthropic/claude-4.5-sonnet-20250514",
            "claude_opus_4_5": "anthropic/claude-opus-4-5-20251101",
            "claude_4_6": "claude-sonnet-4-6",
            "claude_opus": "claude-opus-4-6",
        }
        litellm_model = _LITELLM_NAMES.get(gateway_model)
        if litellm_model is None:
            raise ValueError(
                f"Unknown model {gateway_model!r}. Available: {list(_LITELLM_NAMES.keys())}"
            )

        super().__init__(
            logs_dir=logs_dir,
            model_name=litellm_model,
            temperature=temperature,
            **kwargs,
        )

        # Terminus2.run() 每轮都会用 self._llm 新建 Chat，因此这里直接换掉即可
        self._llm = BedrockClaudeLLM(
            model_name=gateway_model,
            temperature=temperature,
        )

    @staticmethod
    def name() -> str:
        return "bedrock-terminus-2"

    def version(self) -> str | None:
        return "1.0.0"
