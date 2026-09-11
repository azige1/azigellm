"""任务合成公共基础设施。

本模块提供两类被各生成阶段共用的能力：

1. ``batch_chat_completions`` —— 面向本地 vLLM（OpenAI 兼容协议）的并发批量
   对话补全，自带指数退避重试与进度条。
2. ``extract_python_block`` / ``is_valid_python`` —— 从 LLM 输出中抽取 Python
   代码块并做语法校验，供测试生成阶段使用。
"""

from __future__ import annotations

import logging
import random
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from openai import AzureOpenAI, OpenAI, AsyncOpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


MAX_RETRIES = 5


@lru_cache(maxsize=None)
def get_client() -> OpenAI:
    """返回缓存的 OpenAI 兼容客户端（默认指向本机 vLLM 服务）。"""

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="nokey"
    )
    return client


def batch_chat_completions(
    messages: List[List[Dict[str, str]]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    num_completions: int = 1,
    max_concurrency: int = 64,
    show_progress: bool = True,
) -> List[Any]:
    """并发提交一批对话补全请求。

    ``model`` 既可以是模型名（走默认本机端口），也可以是 ``http(s)://`` 开头
    的推理服务端点。返回与输入等长的列表，失败的请求位置为 ``None``。
    """

    # model 参数为 URL 时直接作为服务端点，否则默认本机 8000 端口的 vLLM
    if model and (model.startswith("http://") or model.startswith("https://")):
        base_url = model if model.endswith("/v1") else model.rstrip("/") + "/v1"
    else:
        base_url = "http://localhost:8000/v1"
    client = OpenAI(base_url=base_url, api_key="nokey")
    clients = {
        model: client
    }
    model_keys = list(clients.keys())
    max_retries = MAX_RETRIES

    def _one_with_retry(idx: int, msgs: List[Dict[str, str]]):
        """单个请求：失败时按错误类型选择退避时长后重试。"""
        last_error = None

        for attempt in range(max_retries):
            try:
                model = model_keys[(idx + attempt) % len(model_keys)]
                client = clients[model]

                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=num_completions,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return resp

            except Exception as e:
                last_error = e
                error_str = str(e)

                if attempt < max_retries - 1:
                    # 限流退避更久，超时快速重试，其余走普通指数退避
                    if "rate" in error_str.lower():
                        wait_time = min(2 ** (attempt + 2), 30)
                    elif "timeout" in error_str.lower():
                        wait_time = 2
                    else:
                        wait_time = 2 ** attempt

                    time.sleep(wait_time)
                else:
                    raise last_error

    results: List[Any] = [None] * len(messages)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one_with_retry, i, m): i
            for i, m in enumerate(messages)
        }

        pbar = tqdm(
            total=len(messages),
            disable=False,
            dynamic_ncols=True,
            desc="Processing",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        try:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None
                finally:
                    pbar.update(1)
        finally:
            pbar.close()

    failed_indices = [i for i, r in enumerate(results) if r is None]
    if failed_indices:
        logger.warning(f"Failed requests: {failed_indices}")

    return results


def extract_python_block(code: str) -> str:
    """从 LLM 回复中抽取纯 Python 源码。

    模型输出常带 Markdown 围栏或解释性文字，这里取第一个围栏代码块
    （找不到围栏则把整段视为代码），再去掉公共缩进与尾部空白，
    结果可直接写入 ``.py`` 文件。
    """
    import re
    fence_regex = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_regex.search(code)
    if match:
        snippet = match.group(1)
    else:
        snippet = code
    return textwrap.dedent(snippet).rstrip()  # type: ignore[arg-type]


def is_valid_python(code: str) -> bool:
    """校验一段字符串是否为可编译的 Python 代码。"""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        return False
