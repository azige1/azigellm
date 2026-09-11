"""企业 LLM 网关接入层（Bedrock 兼容 converse 协议）。

通过网关 SDK 建立到 Claude 部署的会话。注意：运行前需先在环境中保存好
网关配置（鉴权信息），并通过环境变量提供各模型的 deployment id。
"""
import enum
import threading

import requests

try:  # 网关 SDK 仅在真实调用时需要；缺失时保持可导入（测试中可注入 Session）
    from gen_ai_hub.proxy.native.amazon.clients import Session
except ImportError:  # pragma: no cover
    Session = None  # type: ignore[assignment]

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

REQUEST_TIMEOUT_SEC = 180

load_dotenv()

DEPLOYMENT_ID_OPUS_4_5 = os.getenv("DEPLOYMENT_ID_OPUS_4_5")
DEPLOYMENT_ID_SONNET_4_5 = os.getenv("DEPLOYMENT_ID_SONNET_4_5")
DEPLOYMENT_ID_OPUS = os.getenv("CLAUDE_OPUS_DEPLOYMENT_ID")
DEPLOYMENT_ID_SONNET = os.getenv("CLAUDE_SONNET_DEPLOYMENT_ID")

DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant that designs CAP CDS models and JS handlers. "
                   "Return code in JSON format like {'schema': <code>, 'service': <code>, 'service_js': <code>}, "
                   "do not include irrelevant text."
    },
    {
        "role": "user",
        "content": "Design a CDS model for a book purchasing order management system. "
                   "Entities are like Book, Author, Customer, Order."
    }
]


class ClaudeDeployment(enum.Enum):
    """Claude 模型注册表：别名、网关模型名、deployment id 三元组。"""
    CLAUDE_4_5 = ("claude_4_5", "anthropic--claude-4.5-sonnet", DEPLOYMENT_ID_SONNET_4_5)
    CLAUDE_OPUS_4_5 = ("claude_opus_4_5", "anthropic--claude-4.5-opus", DEPLOYMENT_ID_OPUS_4_5)
    CLAUDE_4_6 = ("claude_4_6", "anthropic--claude-4.6-sonnet", DEPLOYMENT_ID_SONNET)
    CLAUDE_OPUS = ("claude_opus", "anthropic--claude-4.6-opus", DEPLOYMENT_ID_OPUS)

    def __init__(self, alias: str, model_name: str, deployment_id: str):
        self.alias = alias
        self.model_name = model_name
        self.deployment_id = deployment_id

available_model_aliases = [m.alias for m in ClaudeDeployment]


def get_claude_completion(
    messages: list[dict] = DEFAULT_MESSAGES,
    model: str = "claude_4",
    temperature: float = 0.8,
    max_tokens: int = 4000,
    top_p: float = 0.9,
) -> str:
    """通过网关的 Bedrock 兼容接口请求一次对话补全，返回文本。

    超过 ``REQUEST_TIMEOUT_SEC`` 未返回会抛 ``RuntimeError``。
    """
    if model not in available_model_aliases:
        raise ValueError(f"Model name must be one of {available_model_aliases}")

    # converse() 要求 system 消息单独传
    system_messages = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    formatted_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in messages if m["role"] != "system"
    ]

    model_id = next(m.model_name for m in ClaudeDeployment if m.alias == model)
    deployment_id = next(m.deployment_id for m in ClaudeDeployment if m.alias == model)

    try:
        bedrock = Session().client(model_name=model_id, deployment_id=deployment_id)
        converse_kwargs = dict(
            messages=formatted_messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )
        if system_messages:
            converse_kwargs["system"] = system_messages

        result_container = [None]
        error_container = [None]

        def _call():
            try:
                result_container[0] = bedrock.converse(**converse_kwargs)
            except Exception as e:
                error_container[0] = e

        # SDK 调用没有内置超时，包一层线程手动限时
        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=REQUEST_TIMEOUT_SEC)

        if t.is_alive():
            raise RuntimeError(f"Request timed out after {REQUEST_TIMEOUT_SEC}s")
        if error_container[0] is not None:
            raise error_container[0]

        response = result_container[0]
        return response["output"]["message"]["content"][0]["text"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RequestException occurred while fetching Claude completion: {e}") from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while fetching Claude completion: {e}") from e
