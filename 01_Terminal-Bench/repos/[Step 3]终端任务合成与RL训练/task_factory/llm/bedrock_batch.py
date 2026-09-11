"""``task_factory.batch_chat_completions`` 的网关版本。

函数签名与返回结构与 vLLM 版完全一致（每个元素是带
``resp.choices[0].message.content`` 的轻量命名空间，失败为 ``None``），
可作为即插即用的替换，让整条生成流水线改走企业网关而无需改动下游代码。

用法 —— 替换流水线模块中的导入：

    from task_factory.llm.bedrock_batch import batch_chat_completions
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# 允许从仓库根目录或包内任意位置运行
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task_factory.llm.bedrock_access import get_claude_completion, available_model_aliases

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

_GATEWAY_MODELS = set(available_model_aliases)


def _resolve_model(model: str) -> str:
    """未知模型名回退到 claude_4。"""
    if model in _GATEWAY_MODELS:
        return model
    logger.warning("Model %r is not a gateway alias; falling back to 'claude_4'.", model)
    return "claude_4"


def _make_response(content: str) -> SimpleNamespace:
    """把纯文本包成 OpenAI 响应形状：``resp.choices[0].message.content``。"""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def batch_chat_completions(
    messages: List[List[Dict[str, str]]],
    model: str = "claude_4",
    temperature: float = 0.8,
    max_tokens: int = 4000,
    num_completions: int = 1,  # 仅为签名兼容保留；网关一次只返回一条
    max_concurrency: int = 16,
    show_progress: bool = True,
    **_kwargs: Any,
) -> List[Optional[SimpleNamespace]]:
    """并发请求网关的对话补全接口。

    返回与 ``messages`` 对齐的列表：每项为响应命名空间或 ``None``。
    """
    gateway_model = _resolve_model(model)
    results: List[Optional[SimpleNamespace]] = [None] * len(messages)

    def _call_with_retry(idx: int, msgs: List[Dict[str, str]]) -> Optional[SimpleNamespace]:
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                text = get_claude_completion(
                    messages=msgs,
                    model=gateway_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return _make_response(text)
            except Exception as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.debug("Attempt %d failed for request %d: %s. Retrying in %ds.", attempt + 1, idx, exc, wait)
                time.sleep(wait)
        logger.error("All %d attempts failed for request %d: %s", MAX_RETRIES, idx, last_exc)
        return None

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {pool.submit(_call_with_retry, i, m): i for i, m in enumerate(messages)}

        pbar = tqdm(
            total=len(messages),
            disable=not show_progress,
            dynamic_ncols=True,
            desc="Gateway requests",
            unit="req",
            file=sys.stdout,
        )
        try:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.error("Unexpected error for request %d: %s", idx, exc)
                    results[idx] = None
                finally:
                    pbar.update(1)
        finally:
            pbar.close()

    failed = [i for i, r in enumerate(results) if r is None]
    if failed:
        logger.warning("%d / %d requests failed: indices %s", len(failed), len(messages), failed)

    return results
