"""OpenAI 兼容接口的 runner。"""

import os
from time import sleep
from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from codeharvest.llm.config import LLMConfig
from codeharvest.llm.runner_base import BaseLLMRunner
from codeharvest.llm.models import ModelSpec

class OpenAIChatRunner(BaseLLMRunner):
    # 按服务地址缓存客户端：每个进程内惰性创建，runner 本身只保存地址字符串，便于多进程传参
    _clients: dict[str, OpenAI] = {}

    @staticmethod
    def get_client(base_url: str | None) -> OpenAI:
        key = base_url or ""
        if key not in OpenAIChatRunner._clients:
            OpenAIChatRunner._clients[key] = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"),
                base_url=base_url or None,
            )
        return OpenAIChatRunner._clients[key]

    def __init__(self, args: LLMConfig, model: ModelSpec):
        super().__init__(args, model)
        self.base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
        if "o1" in args.model_name or "o3" in args.model_name or "o4" in args.model_name:
            self.client_kwargs: dict[str, Any] = {
                "model": args.model_name,
                "max_completion_tokens": args.max_tokens,
            }
        else:
            self.client_kwargs = {
                "model": args.model_name,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "n": args.n,
                "timeout": args.openai_timeout,
            }

        # Qwen3 系列关闭 thinking 模式
        if "qwen3" in args.model_name.lower():
            self.client_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    def config(self):
        return self.client_kwargs

    def _run_single(self, payload: list[dict[str, str]]) -> list[str]:
        assert isinstance(payload, list)

        try:
            client = OpenAIChatRunner.get_client(self.base_url)
            response: ChatCompletion = client.chat.completions.create(
                messages=payload,  
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(payload)
        except Exception as e:
            print(f"Failed to run the model for {payload}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]  
