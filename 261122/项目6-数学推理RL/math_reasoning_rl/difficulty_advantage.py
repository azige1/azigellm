"""Group pass-rate peak weighted advantage estimator grouped by uid (6.3)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from math_reasoning_rl.config import PassRateWeightConfig


def compute_pass_rate_weight(
    pass_rate: float | torch.Tensor,
    center: float = 0.25,
    sigma: float = 0.15,
    amplitude: float = 0.6,
) -> float | torch.Tensor:
    """Compute Gaussian-peak difficulty weight ``w(p) = 1 + amplitude * exp(-(p - center)^2 / (2 * sigma^2))``."""
    if isinstance(pass_rate, torch.Tensor):
        return 1.0 + amplitude * torch.exp(-((pass_rate - center) ** 2) / (2.0 * (sigma**2)))
    return 1.0 + amplitude * math.exp(-((pass_rate - center) ** 2) / (2.0 * (sigma**2)))


def compute_pass_rate_weighted_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | list[str],
    config: Optional[PassRateWeightConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute group-centered, unstandardized advantage weighted by group pass-rate peak.

    For each prompt group (grouped by ``uid`` in ``index``):
    - ``R_i = sum_t token_level_rewards[i, t]``
    - ``p = mean(R_i > positive_reward_threshold)``
    - ``w(p) = 1 + 0.6 * exp(-(p - 0.25)^2 / (2 * 0.15^2))`` (when enabled)
    - ``A_i = (R_i - mean(R_group)) * w(p)`` (no std division)
    - Broadcast ``A_i`` over valid tokens via ``response_mask``.
    """
    cfg = config or PassRateWeightConfig()
    scores = token_level_rewards.sum(dim=-1)
    valid_seq = response_mask.sum(dim=-1) > 0

    id2indices: dict[str, list[int]] = defaultdict(list)
    for i in range(scores.shape[0]):
        if bool(valid_seq[i].item()):
            id2indices[str(index[i])].append(i)

    advantages_1d = torch.zeros_like(scores)
    pass_rates: list[float] = []
    weights: list[float] = []
    mid_diff_count = 0

    with torch.no_grad():
        for _uid, row_indices in id2indices.items():
            if len(row_indices) <= 1:
                continue
            group_scores = scores[row_indices]
            group_mean = group_scores.mean()
            pass_rate = float((group_scores > cfg.positive_reward_threshold).float().mean().item())
            weight = (
                float(
                    compute_pass_rate_weight(
                        pass_rate,
                        center=cfg.center,
                        sigma=cfg.sigma,
                        amplitude=cfg.amplitude,
                    )
                )
                if cfg.enabled
                else 1.0
            )
            group_adv = (group_scores - group_mean) * weight
            for pos, row_idx in enumerate(row_indices):
                advantages_1d[row_idx] = group_adv[pos]

            pass_rates.append(pass_rate)
            weights.append(weight)
            if 0.125 <= pass_rate <= 0.375:
                mid_diff_count += 1

        advantages = advantages_1d.unsqueeze(-1) * response_mask.to(dtype=advantages_1d.dtype)

    n_groups = len(pass_rates)
    metrics = {
        "adv/pass_rate_mean": float(np.mean(pass_rates)) if n_groups else 0.0,
        "adv/group_weight_mean": float(np.mean(weights)) if n_groups else 1.0,
        "adv/group_weight_max": float(np.max(weights)) if n_groups else 1.0,
        "adv/mid_difficulty_group_frac": (mid_diff_count / n_groups) if n_groups else 0.0,
    }
    return advantages, advantages, metrics
