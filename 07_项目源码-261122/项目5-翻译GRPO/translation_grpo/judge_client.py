"""Async OpenAI-compatible judge client with concurrency control, retry, and caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

logger = logging.getLogger(__name__)

# Note: Prompt templates use placeholder replacement via str.replace() instead of
# str.format() to avoid KeyError on literal JSON braces inside the templates.
POINTWISE_MQM_PROMPT_TEMPLATE = """You are a professional bilingual translation quality evaluator following the MQM framework.
Evaluate the candidate translation against the source sentence for translation direction __DIRECTION__.

Evaluate errors strictly across these five dimensions:
- accuracy: mistranslation, omission, addition, untranslated text
- terminology: incorrect or inconsistent domain terminology / proper nouns
- fluency: grammar, spelling, punctuation, awkward phrasing
- style: register, tone, formality mismatch
- context: locale conventions, cultural adaptation, contextual coherence

Assign each error span a severity:
- critical: renders translation unusable or reverses core meaning
- major: significantly impairs comprehension or misstates key facts/terms
- minor: slight awkwardness or minor stylistic/punctuation issue

Source:
__SOURCE__

Candidate Translation:
__CANDIDATE__

Return ONLY a valid JSON object with the exact schema:
{
  "errors": [
    {"span": "<erroneous text span>", "dim": "<accuracy|terminology|fluency|style|context>", "severity": "<critical|major|minor>"}
  ]
}
If there are no errors, return {"errors": []}.
"""

DIRECT_SCORE_PROMPT_TEMPLATE = """You are a bilingual translation evaluator for direction __DIRECTION__.
Score the candidate translation from 0 to 100 based on overall translation quality.

Source:
__SOURCE__

Candidate Translation:
__CANDIDATE__

Return ONLY a valid JSON object:
{"score": <number between 0 and 100>}
"""

PAIRWISE_JUDGE_PROMPT_TEMPLATE = """You are a professional bilingual translation quality evaluator for direction __DIRECTION__.
Compare Translation A and Translation B against the source sentence, paying close attention to terminology accuracy, faithfulness, fluency, style, and contextual appropriateness.

Source:
__SOURCE__

Translation A:
__CANDIDATE_A__

Translation B:
__CANDIDATE_B__

Choose which translation is better overall. If they are of essentially equal quality, choose "tie".
Return ONLY a valid JSON object:
{"winner": "<A|B|tie>", "reason": "<brief justification>"}
"""


class JudgeOutputError(RuntimeError):
    """judge 多次返回结构不合格的输出。"""


def render_prompt_template(template: str, replacements: dict[str, str]) -> str:
    """Safely render prompt templates containing JSON braces without str.format()."""
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, str(value))
    return rendered


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {}

    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: scan balanced braces to recover embedded JSON objects.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
        start = cleaned.find("{", start + 1)

    return {}


@dataclass
class JudgeClientConfig:
    """Configuration for OpenAI-compatible LLM judge clients."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    api_key_env: str = "JUDGE_API_KEY"
    model: str = "Qwen3-32B"
    timeout_s: float = 60.0
    max_retries: int = 4
    retry_base_delay: float = 1.0
    retry_max_delay: float = 16.0
    max_concurrency: int = 32
    temperature: float = 0.0
    max_tokens: int = 1024
    cache_path: str | None = None
    enable_cache: bool = True

    def resolve_api_key(self) -> str:
        """Resolve API key from explicit value or environment variable."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            env_val = os.getenv(self.api_key_env, "")
            if env_val:
                return env_val
        return os.getenv("OPENAI_API_KEY", "EMPTY")


TransportFn = Callable[[dict[str, Any]], Awaitable[str]]


class AsyncOpenAIJudgeClient:
    """Async OpenAI-compatible chat completions client with semaphore, retry, and cache."""

    def __init__(
        self,
        config: JudgeClientConfig | None = None,
        transport: TransportFn | None = None,
    ):
        self.config = config or JudgeClientConfig()
        self._transport = transport
        self._semaphore = asyncio.Semaphore(max(1, int(self.config.max_concurrency)))
        self._memory_cache: dict[str, str] = {}
        self._inflight: dict[str, asyncio.Future[str]] = {}
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._cache_file: Path | None = Path(self.config.cache_path) if self.config.cache_path else None
        if self.config.enable_cache and self._cache_file is not None:
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        if self._cache_file is None or not self._cache_file.exists():
            return
        with self._cache_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("key")
                    content = record.get("content")
                    if isinstance(key, str) and isinstance(content, str):
                        self._memory_cache[key] = content
                except json.JSONDecodeError:
                    continue

    def _append_disk_cache(self, key: str, content: str) -> None:
        if not self.config.enable_cache or self._cache_file is None:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")

    @staticmethod
    def _make_cache_key(
        model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int | None = None
    ) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        if self._transport is not None:
            return await self._transport(payload)

        base_url = self.config.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.resolve_api_key()}",
        }
        session = await self._get_session()
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 429 or resp.status >= 500:
                body = await resp.text()
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=body[:500],
                    headers=resp.headers,
                )
            resp.raise_for_status()
            data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Empty choices in judge response: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return content

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:
        last_error: Exception | None = None
        max_attempts = max(1, int(self.config.max_retries))
        for attempt in range(max_attempts):
            try:
                async with self._semaphore:
                    return await asyncio.wait_for(
                        self._post_chat_completion(payload),
                        timeout=self.config.timeout_s,
                    )
            except aiohttp.ClientResponseError as exc:
                if 400 <= exc.status < 500 and exc.status != 429:
                    raise
                last_error = exc
            except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as exc:
                last_error = exc

            if attempt < max_attempts - 1:
                delay = min(
                    self.config.retry_base_delay * (2**attempt),
                    self.config.retry_max_delay,
                )
                logger.warning(
                    "Judge request failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt + 1,
                    max_attempts,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    async def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request with deduplicated caching and retries."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        eff_temp = self.config.temperature if temperature is None else float(temperature)
        eff_max_tokens = self.config.max_tokens if max_tokens is None else int(max_tokens)
        cache_key = self._make_cache_key(self.config.model, messages, eff_temp, eff_max_tokens)

        if self.config.enable_cache:
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]

            async with self._lock:
                if cache_key in self._memory_cache:
                    return self._memory_cache[cache_key]
                existing_future = self._inflight.get(cache_key)
                if existing_future is None:
                    loop = asyncio.get_running_loop()
                    existing_future = loop.create_future()
                    self._inflight[cache_key] = existing_future
                    is_owner = True
                else:
                    is_owner = False

            if not is_owner:
                return await existing_future

            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": eff_temp,
                "max_tokens": eff_max_tokens,
            }
            try:
                content = await self._call_with_retry(payload)
                async with self._lock:
                    self._memory_cache[cache_key] = content
                    self._append_disk_cache(cache_key, content)
                    fut = self._inflight.pop(cache_key, None)
                    if fut is not None and not fut.done():
                        fut.set_result(content)
                return content
            except BaseException as exc:
                # 含取消：清理在途记录，并把失败或取消传给等待同一请求的协程
                async with self._lock:
                    fut = self._inflight.pop(cache_key, None)
                    if fut is not None and not fut.done():
                        if isinstance(exc, asyncio.CancelledError):
                            fut.cancel()
                        else:
                            fut.set_exception(exc)
                raise

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
        }
        return await self._call_with_retry(payload)

    async def invalidate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """从内存缓存中移除一条响应（结构不合格的输出不能被后续请求复用）。"""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        eff_temp = self.config.temperature if temperature is None else float(temperature)
        eff_max_tokens = self.config.max_tokens if max_tokens is None else int(max_tokens)
        key = self._make_cache_key(self.config.model, messages, eff_temp, eff_max_tokens)
        async with self._lock:
            self._memory_cache.pop(key, None)

    async def chat_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        validator: Callable[[dict[str, Any]], bool] | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any], str]:
        """Call chat() and parse JSON object from the completion.

        给定 ``validator`` 时，结构不合格的输出会从缓存移除并重新请求；
        连续 ``max_attempts`` 次不合格抛出 ``JudgeOutputError``。
        """
        raw = ""
        for _ in range(max(1, int(max_attempts))):
            raw = await self.chat(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = extract_json_object(raw)
            if validator is None or validator(parsed):
                return parsed, raw
            await self.invalidate(
                prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
            )
        raise JudgeOutputError(f"judge 输出结构不合格（已重试 {max_attempts} 次）: {raw[:200]}")
