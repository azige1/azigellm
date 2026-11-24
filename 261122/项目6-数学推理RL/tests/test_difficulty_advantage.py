"""Unit tests for 6.3: uid-grouped pass-rate peak weighted advantage."""

from __future__ import annotations

import math

import numpy as np
import torch

from math_reasoning_rl.config import PassRateWeightConfig
from math_reasoning_rl.difficulty_advantage import compute_pass_rate_weight, compute_pass_rate_weighted_advantage


def test_pass_rate_weight_values():
    w_peak = compute_pass_rate_weight(0.25, center=0.25, sigma=0.15, amplitude=0.6)
    assert abs(w_peak - 1.6) < 1e-9

    w_one_eighth = compute_pass_rate_weight(0.125, center=0.25, sigma=0.15, amplitude=0.6)
    expected = 1.0 + 0.6 * math.exp(-((0.125 - 0.25) ** 2) / (2.0 * 0.15**2))
    assert abs(w_one_eighth - expected) < 1e-9
    assert abs(w_one_eighth - 1.4243) < 1e-3

    w_high = compute_pass_rate_weight(0.75, center=0.25, sigma=0.15, amplitude=0.6)
    assert w_high < 1.01


def test_uid_grouped_advantage_without_std_division():
    # Group A (8 trajectories): 2 pass (+1), 6 fail (-1) -> p = 0.25, w = 1.6, mean = -0.5
    # Correct adv = (1 - (-0.5)) * 1.6 = 2.4; Wrong adv = (-1 - (-0.5)) * 1.6 = -0.8
    # Group B (4 trajectories): 3 pass (+1), 1 fail (-1) -> p = 0.75, w ~= 1.0023, mean = +0.5
    # Padding row (response_mask = 0) should be ignored and have 0 advantage.
    rewards = torch.zeros((13, 3), dtype=torch.float32)
    rewards[:2, -1] = 1.0
    rewards[2:8, -1] = -1.0
    rewards[8:11, -1] = 1.0
    rewards[11, -1] = -1.0
    rewards[12, -1] = 999.0  # padding row

    mask = torch.ones((13, 3), dtype=torch.float32)
    mask[:, 2] = 0.0
    mask[12, :] = 0.0

    uids = np.array(["gA"] * 8 + ["gB"] * 4 + ["pad"], dtype=object)

    adv, ret, metrics = compute_pass_rate_weighted_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=uids,
        config=PassRateWeightConfig(center=0.25, sigma=0.15, amplitude=0.6),
    )

    assert torch.allclose(adv, ret)
    assert torch.allclose(adv[0, :2], torch.tensor([2.4, 2.4]), atol=1e-5)
    assert torch.allclose(adv[2, :2], torch.tensor([-0.8, -0.8]), atol=1e-5)
    assert torch.allclose(adv[:, 2], torch.zeros(13))
    assert torch.allclose(adv[12], torch.zeros(3))

    # Group B should NOT contaminate Group A, and Group B uses its own p = 0.75.
    w_b = compute_pass_rate_weight(0.75)
    assert torch.allclose(adv[8, :2], torch.full((2,), (1.0 - 0.5) * w_b), atol=1e-5)
    assert torch.allclose(adv[11, :2], torch.full((2,), (-1.0 - 0.5) * w_b), atol=1e-5)
    assert abs(metrics["adv/mid_difficulty_group_frac"] - 0.5) < 1e-9
