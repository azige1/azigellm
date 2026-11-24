"""Unit tests for 6.2: entropy bandwidth hinge regularization and setup hook patching."""

from __future__ import annotations

import sys
import types

import torch

from math_reasoning_rl.entropy_band_loss import compute_entropy_band_hinge_loss, entropy_band_ppo_loss
from math_reasoning_rl.runtime_hooks import apply_patch_if_loaded, setup_worker_patches


def test_entropy_band_hinge_inside_and_outside():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    # Inside [0.45, 0.65] -> exact zero loss and zero gradient.
    ent_in = torch.full((2, 4), 0.55, requires_grad=True)
    loss_in, m_in = compute_entropy_band_hinge_loss(ent_in, mask, low=0.45, high=0.65, coef=0.005)
    assert abs(loss_in.item()) < 1e-9
    assert m_in["actor/entropy_band_in_range"] == 1.0
    loss_in.backward()
    assert torch.allclose(ent_in.grad, torch.zeros_like(ent_in))

    # Above upper bound: H = 0.71 -> 0.005 * (0.71 - 0.65) = 0.0003, positive grad wrt entropy.
    ent_hi = torch.full((2, 4), 0.71, requires_grad=True)
    loss_hi, m_hi = compute_entropy_band_hinge_loss(ent_hi, mask, low=0.45, high=0.65, coef=0.005)
    assert abs(loss_hi.item() - 0.0003) < 1e-6
    assert abs(m_hi["actor/entropy_band_upper_violation"] - 0.06) < 1e-6
    assert m_hi["actor/entropy_band_in_range"] == 0.0
    loss_hi.backward()
    assert (ent_hi.grad[mask] > 0).all()
    assert torch.allclose(ent_hi.grad[~mask], torch.zeros_like(ent_hi.grad[~mask]))

    # Below lower bound: H = 0.35 -> 0.005 * (0.45 - 0.35) = 0.0005, negative grad wrt entropy.
    ent_lo = torch.full((2, 4), 0.35, requires_grad=True)
    loss_lo, m_lo = compute_entropy_band_hinge_loss(ent_lo, mask, low=0.45, high=0.65, coef=0.005)
    assert abs(loss_lo.item() - 0.0005) < 1e-6
    assert abs(m_lo["actor/entropy_band_lower_violation"] - 0.10) < 1e-6
    loss_lo.backward()
    assert (ent_lo.grad[mask] < 0).all()


def test_entropy_band_microbatch_token_share_scaling():
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    ent = torch.full((1, 4), 0.75, requires_grad=True)
    # local_tokens = 2, global batch_num_tokens = 8, dp_size = 2 -> share = 2 / 8 * 2 = 0.5
    loss, _ = compute_entropy_band_hinge_loss(
        ent, mask, low=0.45, high=0.65, coef=0.005, dp_size=2, batch_num_tokens=8
    )
    expected = 0.005 * 0.10 * 0.5
    assert abs(loss.item() - expected) < 1e-7


def test_setup_worker_patches_does_not_eagerly_import_and_patches_targets(monkeypatch):
    # Create fake verl.workers.utils.losses and verl.workers.engine_workers in sys.modules
    # to verify both bindings are replaced cleanly.
    fake_losses = types.ModuleType("verl.workers.utils.losses")
    orig_fn = lambda config, model_output, data, dp_group=None: (torch.tensor(0.0), {})  # noqa: E731
    fake_losses.ppo_loss = orig_fn

    fake_engine_workers = types.ModuleType("verl.workers.engine_workers")
    fake_engine_workers.ppo_loss = orig_fn

    monkeypatch.setitem(sys.modules, "verl.workers.utils.losses", fake_losses)
    monkeypatch.setitem(sys.modules, "verl.workers.engine_workers", fake_engine_workers)

    setup_worker_patches()
    apply_patch_if_loaded()

    assert fake_losses.ppo_loss is entropy_band_ppo_loss
    assert fake_engine_workers.ppo_loss is entropy_band_ppo_loss
    assert getattr(fake_losses, "_orig_ppo_loss") is orig_fn
