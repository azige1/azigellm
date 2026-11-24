"""Unit tests for 6.4: single-step overflow reuse buffer and truncated importance ratio."""

from __future__ import annotations

import sys
import types

import torch

from math_reasoning_rl.overflow_replay_buffer import compute_truncated_importance_weights


def test_truncated_importance_weights_clips_reused_only():
    old_lp = torch.tensor(
        [
            [-0.1, -0.2, -5.0],  # reused row: ratio exp(2.0) > 2 -> clipped to 2.0, exp(0.1) ~= 1.105
            [-1.0, -0.5, -5.0],  # fresh row: forced to 1.0
        ],
        dtype=torch.float32,
    )
    sample_lp = torch.tensor(
        [
            [-2.1, -0.3, -5.0],
            [-3.5, -0.1, -5.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
    is_reused = torch.tensor([True, False], dtype=torch.bool)

    weights, metrics = compute_truncated_importance_weights(
        old_log_probs=old_lp,
        sampling_log_probs=sample_lp,
        response_mask=mask,
        is_reused_mask=is_reused,
        clip_max=2.0,
    )

    assert abs(weights[0, 0].item() - 2.0) < 1e-6
    assert abs(weights[0, 1].item() - torch.exp(torch.tensor(0.1)).item()) < 1e-5
    assert weights[0, 2].item() == 0.0
    assert torch.allclose(weights[1], torch.tensor([1.0, 1.0, 0.0]))
    assert abs(metrics["training/overflow_buffer/reused_is_clipped_frac"] - 0.5) < 1e-6
    assert metrics["training/overflow_buffer/reused_is_ratio_raw_max"] > 7.0


def test_overflow_buffer_single_step_ttl_and_pause_gating(monkeypatch):
    # Provide lightweight stub for transfer_queue if verl/transfer_queue is unavailable in CPU test env.
    if "transfer_queue" not in sys.modules:
        fake_tq = types.ModuleType("transfer_queue")
        fake_tq.KVBatchMeta = type("KVBatchMeta", (), {})
        fake_tq.kv_clear = lambda **kwargs: None
        fake_tq.kv_list = lambda *args, **kwargs: {}
        fake_tq.kv_batch_get = lambda **kwargs: {}
        monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)

    try:
        from omegaconf import OmegaConf
        from math_reasoning_rl.overflow_replay_buffer import OverflowReuseReplayBuffer
    except Exception:
        # If verl is not installed in the unit test environment, skip ReplayBuffer subclass test cleanly.
        return

    cleared_calls: list[set[str]] = []
    buf = OverflowReuseReplayBuffer(
        trainer_mode="sync",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={
            "filter_groups_metric": "reward",
            "train_batch_size": 4,
            "gen_batch_size": 1,
            "max_inflight_gen_batches": 2,
            "max_reuse_steps": 1,
            "max_reuse_ratio": 0.5,
        },
        refill_fn=lambda n: None,
    )
    monkeypatch.setattr(buf, "_clear_groups", lambda partition_id, uids: cleared_calls.append(set(uids)))

    # Buffer 2 overflow groups from step 1.
    buf._overflow_buffer["train"] = {"u1": 1, "u2": 1}
    assert buf.get_reusable_count(global_steps=2) == 2
    # Step 3 exceeds max_reuse_steps=1 -> expired.
    assert buf.get_reusable_count(global_steps=3) == 0

    expired, paused = buf._evict_expired_overflow(global_steps=3, partition_id="train")
    assert expired == 2
    assert paused is False
    assert {"u1", "u2"} in cleared_calls

    # When previous step reused_ratio > 0.5, next step pauses reuse and clears buffer.
    buf._overflow_buffer["train"] = {"u3": 3}
    buf._pause_next_reuse["train"] = True
    assert buf.get_reusable_count(global_steps=4) == 0
    expired2, paused2 = buf._evict_expired_overflow(global_steps=4, partition_id="train")
    assert paused2 is True
    assert expired2 == 1
    assert buf._pause_next_reuse["train"] is False
