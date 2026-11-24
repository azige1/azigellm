"""Unit tests for 5.3 anchor-corrected GRPO advantage and 5.4 hard mask + zero LoRA grad."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from translation_grpo.advantage_mask import (
    AnchorAdvantageConfig,
    compute_translation_grpo_advantage,
    verify_zero_grad_on_masked_batch,
)


def test_anchor_advantage_numerical_alignment_and_isolation():
    # Group 0: 4 candidates, all poor relative to anchor (-1.0)
    # Group 1: 4 candidates, high quality around anchor (-2.0)
    rewards = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, -1.5, -2.5, -3.0], dtype=torch.float32)
    anchor_rewards = torch.tensor([-1.0] * 4 + [-2.0] * 4, dtype=torch.float32)
    uids = ["g0"] * 4 + ["g1"] * 4
    response_mask = torch.ones((8, 6), dtype=torch.int64)

    out_no_anc = compute_translation_grpo_advantage(
        rewards,
        response_mask,
        uids,
        config=AnchorAdvantageConfig(anchor_enabled=False, hard_mask_enabled=True),
    )
    out_with_anc = compute_translation_grpo_advantage(
        rewards,
        response_mask,
        uids,
        anchor_rewards=anchor_rewards,
        config=AnchorAdvantageConfig(anchor_enabled=True, anchor_lambda=0.3, hard_mask_enabled=True),
    )

    # Best candidate in below-anchor group (idx 3: r=-2.0 < r_anc=-1.0) has positive advantage damped.
    assert out_with_anc["scalar_advantages"][3] < out_no_anc["scalar_advantages"][3]
    # Candidate beating anchor (idx 4: r=-1.0 > r_anc=-2.0) gets positive anchor boost.
    assert out_with_anc["scalar_advantages"][4] > out_no_anc["scalar_advantages"][4]
    # Anchor delta is strictly bounded within (-lambda, +lambda).
    delta = (out_with_anc["scalar_advantages"] - out_no_anc["scalar_advantages"]).abs()
    assert float(delta.max().item()) < 0.3 + 1e-6


def test_anchor_row_inside_batch_is_masked_from_stats_and_loss():
    rewards = torch.tensor([-4.0, -2.0, -3.0, -1.0], dtype=torch.float32)
    uids = ["g0", "g0", "g0", "g0"]
    response_mask = torch.ones((4, 5), dtype=torch.int64)
    is_anchor_mask = torch.tensor([False, False, False, True], dtype=torch.bool)

    out = compute_translation_grpo_advantage(
        rewards,
        response_mask,
        uids,
        is_anchor_mask=is_anchor_mask,
        config=AnchorAdvantageConfig(anchor_enabled=True, anchor_lambda=0.3),
    )
    assert out["response_mask"][3].sum().item() == 0
    assert out["loss_mask"][3].sum().item() == 0
    assert out["advantages"][3].abs().sum().item() == 0.0


def test_hard_mask_excludes_invalid_from_group_stats_and_skips_small_groups():
    # Group 0 has 1 invalid sample with huge negative score -50.0 that must NOT inflate valid peers.
    # Group 1 has only 1 valid sample left (< 2), so the whole group must be skipped.
    rewards = torch.tensor([-2.0, -4.0, -3.0, -50.0, -2.0, -10.0], dtype=torch.float32)
    uids = ["g0", "g0", "g0", "g0", "g1", "g1"]
    invalid_mask = torch.tensor([False, False, False, True, False, True], dtype=torch.bool)
    response_mask = torch.ones((6, 4), dtype=torch.int64)

    out = compute_translation_grpo_advantage(
        rewards,
        response_mask,
        uids,
        invalid_mask=invalid_mask,
        config=AnchorAdvantageConfig(anchor_enabled=False, hard_mask_enabled=True, min_valid_group_size=2),
    )

    # Group 0 valid rewards are [-2, -4, -3] -> mean = -3, std = 1.0 -> adv = [+1, -1, 0].
    assert out["scalar_advantages"][0].item() == pytest.approx(1.0, abs=1e-4)
    assert out["scalar_advantages"][1].item() == pytest.approx(-1.0, abs=1e-4)
    assert out["scalar_advantages"][2].item() == pytest.approx(0.0, abs=1e-4)

    # Invalid sample (idx 3) and skipped group g1 (idx 4, 5) are completely masked out.
    for masked_idx in (3, 4, 5):
        assert out["response_mask"][masked_idx].sum().item() == 0
        assert out["loss_mask"][masked_idx].sum().item() == 0
        assert out["advantages"][masked_idx].abs().sum().item() == 0.0
    assert out["metrics"]["grpo/skipped_group_count"] == pytest.approx(1.0)


def test_padding_samples_do_not_pollute_group_stats():
    rewards = torch.tensor([-2.0, -4.0, 0.0], dtype=torch.float32)
    uids = ["g0", "g0", "g0"]
    is_padding = torch.tensor([False, False, True], dtype=torch.bool)
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=torch.int64)

    out = compute_translation_grpo_advantage(
        rewards,
        response_mask,
        uids,
        is_padding_mask=is_padding,
        config=AnchorAdvantageConfig(anchor_enabled=False),
    )
    # Valid rewards [-2, -4] -> mean = -3, std = sqrt(2) -> +0.7071 / -0.7071
    expected = 1.0 / math.sqrt(2.0)
    assert out["scalar_advantages"][0].item() == pytest.approx(expected, abs=1e-4)
    assert out["scalar_advantages"][1].item() == pytest.approx(-expected, abs=1e-4)
    assert out["response_mask"][2].sum().item() == 0


class _TinyLoRAModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lora_a = nn.Linear(hidden, 4, bias=False)
        self.lora_b = nn.Linear(4, hidden, bias=False)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        h = self.embed(input_ids)
        h = h + self.lora_b(self.lora_a(h))
        return (self.head(h),)


def test_verify_zero_grad_on_masked_batch():
    model = _TinyLoRAModel()
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.int64)
    old_log_probs = torch.zeros((2, 4), dtype=torch.float32)
    advantages = torch.zeros((2, 4), dtype=torch.float32)
    response_mask = torch.zeros((2, 4), dtype=torch.int64)

    grad_norm = verify_zero_grad_on_masked_batch(
        model,
        input_ids,
        attention_mask,
        old_log_probs,
        advantages,
        response_mask,
        param_filter=lambda name, _: "lora_" in name,
    )
    assert grad_norm == pytest.approx(0.0)
