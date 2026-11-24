import pytest
import torch

from multi_teacher_distillation.domain_advantage import DomainAdvantageNormalizer


def test_domain_running_std_normalization_balances_domain_scales():
    normalizer = DomainAdvantageNormalizer(decay=0.9, sigma_floor=0.2, raw_clip=5.0, post_norm_clip=3.0)
    # Row 0: issue_fix (large scale std ~ 2.0); Row 1: instruction_following (small scale std ~ 0.4)
    # Row 2: self_anchor (std ~ 0.05 -> clamped by floor 0.2)
    raw = torch.tensor([
        [4.0, -4.0, 2.0, -2.0],
        [0.4, -0.4, 0.4, -0.4],
        [0.05, -0.05, 0.0, 0.0],
    ])
    mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ])
    domains = ["issue_fix", "instruction_following", "self_anchor"]

    normed, diag = normalizer.normalize_batch(raw, mask, domains)
    assert diag["effective_sigmas"]["issue_fix"] == pytest.approx(3.1622776, abs=1e-4)
    assert diag["effective_sigmas"]["instruction_following"] == pytest.approx(0.4, abs=1e-4)
    assert diag["effective_sigmas"]["self_anchor"] == pytest.approx(0.2, abs=1e-4)
    assert normed[1, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert normed[2, 2].item() == pytest.approx(0.0)

    # Second step checks EMA decay 0.9 * prev + 0.1 * current
    raw2 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    _, diag2 = normalizer.normalize_batch(raw2, mask, domains)
    assert diag2["running_stds"]["instruction_following"] == pytest.approx(0.9 * 0.4 + 0.1 * 1.0, abs=1e-4)


def test_raw_clip_five_and_post_norm_clip_three():
    normalizer = DomainAdvantageNormalizer(decay=0.9, sigma_floor=0.2, raw_clip=5.0, post_norm_clip=3.0)
    raw = torch.tensor([[10.0, -10.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1]])
    normed, _ = normalizer.normalize_batch(raw, mask, ["code_gen"])
    assert normed.max().item() <= 3.0
    assert normed.min().item() >= -3.0
