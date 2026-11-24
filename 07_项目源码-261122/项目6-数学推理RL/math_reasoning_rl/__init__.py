"""Mathematical reasoning RL package built on verl 0.9.1 V1."""

from math_reasoning_rl.config import (
    EntropyBandConfig,
    OverflowReuseConfig,
    PassRateWeightConfig,
    RewardRoutingConfig,
)
from math_reasoning_rl.difficulty_advantage import (
    compute_pass_rate_weight,
    compute_pass_rate_weighted_advantage,
)
from math_reasoning_rl.entropy_band_loss import (
    compute_entropy_band_hinge_loss,
    entropy_band_ppo_loss,
)

__all__ = [
    "EntropyBandConfig",
    "OverflowReuseConfig",
    "PassRateWeightConfig",
    "RewardRoutingConfig",
    "compute_entropy_band_hinge_loss",
    "entropy_band_ppo_loss",
    "compute_pass_rate_weight",
    "compute_pass_rate_weighted_advantage",
]
