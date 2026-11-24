import asyncio
import hashlib
from collections import defaultdict
from typing import Any

import httpx

from .schemas import ScoreRequest, ScoreResponse


class TeacherPoolClient:
    def __init__(
        self,
        endpoints: list[str],
        teacher_versions: dict[str, str],
        timeout: float = 1800.0,
        max_attempts: int = 4,
    ) -> None:
        if not endpoints:
            raise ValueError("教师池至少需要一个 endpoint")
        self.endpoints = [value.rstrip("/") for value in endpoints]
        self.teacher_versions = dict(teacher_versions)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.clients: list[httpx.AsyncClient] = []

    def _endpoint(self, teacher_id: str) -> str:
        digest = hashlib.sha256(teacher_id.encode("utf-8")).digest()
        return self.endpoints[int.from_bytes(digest[:4], "big") % len(self.endpoints)]

    async def _client(self, index: int) -> httpx.AsyncClient:
        while len(self.clients) < len(self.endpoints):
            self.clients.append(httpx.AsyncClient(timeout=self.timeout))
        return self.clients[index]

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        index = self.endpoints.index(endpoint)
        for attempt in range(self.max_attempts):
            try:
                client = await self._client(index)
                response = await client.post(endpoint + "/score_batch", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as error:
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(f"教师池请求失败: {endpoint}/score_batch") from error
                await asyncio.sleep(0.25 * 2**attempt)
        raise AssertionError("不可达")

    async def score_batch(
        self,
        requests: list[ScoreRequest],
        strict_version: bool = True,
    ) -> list[ScoreResponse]:
        grouped = defaultdict(list)
        for request in requests:
            grouped[request.teacher_id].append(request)
        responses: dict[str, ScoreResponse] = {}

        async def score_group(teacher_id: str, group: list[ScoreRequest]) -> None:
            payload = await self._post(
                self._endpoint(teacher_id),
                {"requests": [request.model_dump() for request in group]},
            )
            for raw in payload["responses"]:
                response = ScoreResponse.model_validate(raw)
                expected_version = self.teacher_versions.get(teacher_id)
                if strict_version and expected_version is not None and expected_version != response.model_version:
                    raise RuntimeError(
                        f"教师 {teacher_id} 的版本与运行配置不一致: "
                        f"expected={expected_version}, got={response.model_version}"
                    )
                responses[response.request_id] = response

        await asyncio.gather(*(score_group(name, group) for name, group in grouped.items()))
        return [responses[request.request_id] for request in requests]

    async def close(self) -> None:
        await asyncio.gather(*(client.aclose() for client in self.clients))
        self.clients = []
