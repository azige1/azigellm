"""Harbor ``BaseLLM`` 的自定义实现：把 LLM 调用改道到企业网关的
Bedrock 兼容接口，替代默认的 LiteLLM 后端。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from gen_ai_hub.proxy.native.amazon.clients import Session

from harbor.llms.base import BaseLLM, LLMResponse, OutputLengthExceededError
from harbor.models.metric import UsageInfo

from task_factory.llm.bedrock_access import ClaudeDeployment

logger = logging.getLogger(__name__)

# 别名与网关模型名都注册进查询表
_MODEL_MAP: dict[str, ClaudeDeployment] = {}
for m in ClaudeDeployment:
    _MODEL_MAP[m.alias] = m
    _MODEL_MAP[m.model_name] = m


def _resolve_model(model_name: str) -> ClaudeDeployment:
    """把任意模型名字符串解析为注册表条目。"""
    if model_name in _MODEL_MAP:
        return _MODEL_MAP[model_name]
    lower = model_name.lower()
    for key, val in _MODEL_MAP.items():
        if key.lower() == lower:
            return val
    raise ValueError(
        f"Unknown model {model_name!r}. Available models: {list(_MODEL_MAP.keys())}"
    )


class BedrockClaudeLLM(BaseLLM):
    """走 Bedrock 兼容 converse 接口的 Claude 后端。"""

    def __init__(
        self,
        model_name: str = "claude_opus",
        temperature: float = 0.7,
        max_tokens: int = 16384,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_enum = _resolve_model(model_name)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._logger = logger.getChild(self.__class__.__name__)
        self._logger.info(
            "Gateway LLM initialized: %s (deployment=%s)",
            self._model_enum.model_name,
            self._model_enum.deployment_id,
        )

    def _get_bedrock_client(self):
        """每次新建客户端，让 SDK 自行处理令牌刷新。"""
        session = Session()
        return session.client(
            model_name=self._model_enum.model_name,
            deployment_id=self._model_enum.deployment_id,
        )

    def _format_messages(
        self, prompt: str, message_history: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        """把 Harbor 的消息格式转换为 converse() 需要的格式。

        返回 ``(system 消息列表, 对话消息列表)``。
        """
        all_messages = list(message_history) + [{"role": "user", "content": prompt}]

        system_messages = []
        conversation_messages = []

        for msg in all_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_messages.append({"text": content})
            else:
                # converse() 的 content 是内容块列表
                if isinstance(content, str):
                    content_blocks = [{"text": content}]
                elif isinstance(content, list):
                    content_blocks = []
                    for block in content:
                        if isinstance(block, str):
                            content_blocks.append({"text": block})
                        elif isinstance(block, dict) and "text" in block:
                            content_blocks.append({"text": block["text"]})
                        else:
                            content_blocks.append({"text": str(block)})
                else:
                    content_blocks = [{"text": str(content)}]

                conversation_messages.append({
                    "role": role,
                    "content": content_blocks,
                })

        return system_messages, conversation_messages

    def _call_sync(
        self, prompt: str, message_history: list[dict[str, Any]]
    ) -> LLMResponse:
        """同步调用网关 converse 接口。"""
        client = self._get_bedrock_client()
        system_msgs, conv_msgs = self._format_messages(prompt, message_history)

        converse_kwargs: dict[str, Any] = {
            "messages": conv_msgs,
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
            },
        }
        if system_msgs:
            converse_kwargs["system"] = system_msgs

        response = client.converse(**converse_kwargs)

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        content = ""
        for block in content_blocks:
            if "text" in block:
                content += block["text"]

        usage_data = response.get("usage", {})
        prompt_tokens = usage_data.get("inputTokens", 0)
        completion_tokens = usage_data.get("outputTokens", 0)
        # 网关不单独上报 cache token
        cache_tokens = 0

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_tokens=cache_tokens,
            cost_usd=0.0,  # 网关不上报费用
        )

        stop_reason = response.get("stopReason", "end_turn")
        if stop_reason == "max_tokens":
            raise OutputLengthExceededError(
                f"Model hit max_tokens limit ({self._max_tokens}). Response truncated.",
                truncated_response=content,
            )

        return LLMResponse(
            content=content,
            reasoning_content=None,
            model_name=self._model_enum.model_name,
            usage=usage,
        )

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        response_format: Any = None,
        logging_path: Any = None,
        **kwargs,
    ) -> LLMResponse:
        """异步入口：把同步 SDK 调用丢到线程里执行。"""
        if message_history is None:
            message_history = []

        return await asyncio.to_thread(self._call_sync, prompt, message_history)

    def get_model_context_limit(self) -> int:
        """当前 Claude 模型支持 200k 上下文。"""
        return 200000

    def get_model_output_limit(self) -> int | None:
        """当前 Claude 模型最大输出 32k token。"""
        return 32000
