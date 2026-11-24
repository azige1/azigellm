from typing import Any

from pydantic import BaseModel, field_validator


class ScoreRequest(BaseModel):
    request_id: str
    teacher_id: str
    token_ids: list[int]
    positions: list[int]
    top_k: int = 0
    candidate_token_ids: list[list[int]] | None = None
    expected_version: str | None = None

    @field_validator("positions")
    @classmethod
    def positions_are_ordered(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("positions 必须严格递增")
        return value

    @field_validator("top_k")
    @classmethod
    def top_k_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("top_k 不能为负")
        return value


class PositionScore(BaseModel):
    position: int
    sampled_id: int
    sampled_logprob: float
    top_k_ids: list[int] = []
    top_k_logprobs: list[float] = []
    candidate_logprobs: dict[int, float] = {}


class ScoreResponse(BaseModel):
    request_id: str
    teacher_id: str
    scores: list[PositionScore]
    model_version: str
    prefill_tokens: int


class BatchRequest(BaseModel):
    requests: list[dict[str, Any]]
