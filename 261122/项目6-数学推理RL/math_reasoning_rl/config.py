"""Configuration dataclasses for math reasoning RL components."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RewardRoutingConfig:
    """Configuration for gated rule-or-judge reward routing (6.1)."""

    repetition_ngram_n: int = 4
    repetition_threshold: float = 0.3
    require_think_tags: bool = True
    judge_base_url: str = "http://127.0.0.1:8001/v1"
    judge_api_key: str = "EMPTY"
    judge_model: str = "Qwen/Qwen3-32B"
    judge_timeout_s: float = 60.0
    judge_max_retries: int = 3
    judge_temperature: float = 0.0
    judge_max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "RewardRoutingConfig":
        """Build configuration from environment variables when running inside workers."""
        return cls(
            repetition_ngram_n=int(os.getenv("MATH_RL_REP_NGRAM_N", "4")),
            repetition_threshold=float(os.getenv("MATH_RL_REP_THRESHOLD", "0.3")),
            require_think_tags=os.getenv("MATH_RL_REQUIRE_THINK", "1") == "1",
            judge_base_url=os.getenv("MATH_RL_JUDGE_BASE_URL", "http://127.0.0.1:8001/v1"),
            judge_api_key=os.getenv("MATH_RL_JUDGE_API_KEY", "EMPTY"),
            judge_model=os.getenv("MATH_RL_JUDGE_MODEL", "Qwen/Qwen3-32B"),
            judge_timeout_s=float(os.getenv("MATH_RL_JUDGE_TIMEOUT_S", "60.0")),
            judge_max_retries=int(os.getenv("MATH_RL_JUDGE_MAX_RETRIES", "3")),
        )


@dataclass
class EntropyBandConfig:
    """Configuration for entropy bandwidth hinge regularization (6.2)."""

    low: float = 0.45
    high: float = 0.65
    coef: float = 0.005
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "EntropyBandConfig":
        """Read entropy band settings forwarded to Ray worker processes."""
        return cls(
            low=float(os.getenv("MATH_RL_ENTROPY_LOW", "0.45")),
            high=float(os.getenv("MATH_RL_ENTROPY_HIGH", "0.65")),
            coef=float(os.getenv("MATH_RL_ENTROPY_COEF", "0.005")),
            enabled=os.getenv("MATH_RL_ENTROPY_ENABLE", "1") == "1",
        )

    def to_env(self) -> dict[str, str]:
        """Serialize settings into environment variables for Ray runtime_env."""
        return {
            "MATH_RL_ENTROPY_LOW": str(self.low),
            "MATH_RL_ENTROPY_HIGH": str(self.high),
            "MATH_RL_ENTROPY_COEF": str(self.coef),
            "MATH_RL_ENTROPY_ENABLE": "1" if self.enabled else "0",
        }


@dataclass
class PassRateWeightConfig:
    """Configuration for group pass-rate peak weighted advantage (6.3)."""

    center: float = 0.25
    sigma: float = 0.15
    amplitude: float = 0.6
    positive_reward_threshold: float = 0.0
    enabled: bool = True


@dataclass
class OverflowReuseConfig:
    """Configuration for single-step overflow reuse and truncated IS (6.4)."""

    max_reuse_steps: int = 1
    max_reuse_ratio: float = 0.5
    is_ratio_clip: float = 2.0
    enabled: bool = True
