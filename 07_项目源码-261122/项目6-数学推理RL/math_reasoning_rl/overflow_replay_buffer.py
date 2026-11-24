"""Single-step overflow sample reuse replay buffer and truncated IS helpers (6.4)."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Optional

import torch
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, _accumulate_eviction_metrics


def compute_truncated_importance_weights(
    old_log_probs: torch.Tensor,
    sampling_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    is_reused_mask: torch.Tensor,
    clip_max: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute token-level truncated importance ratio ``min(exp(old - sample), clip_max)`` for reused groups.

    Fresh groups (``is_reused_mask == False``) receive weight ``1.0`` on valid response tokens.
    """
    mask_float = response_mask.to(dtype=old_log_probs.dtype)
    reused_row_mask = is_reused_mask.to(device=old_log_probs.device, dtype=torch.bool).unsqueeze(-1)

    log_ratio = torch.clamp(old_log_probs - sampling_log_probs, min=-20.0, max=20.0)
    raw_ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(raw_ratio, max=clip_max)

    weights = torch.where(reused_row_mask, clipped_ratio, torch.ones_like(clipped_ratio))
    weights = (weights * mask_float).detach()

    reused_token_mask = reused_row_mask & response_mask.bool()
    reused_token_count = int(reused_token_mask.sum().item())
    if reused_token_count > 0:
        reused_raw = raw_ratio[reused_token_mask]
        reused_clipped = clipped_ratio[reused_token_mask]
        is_ratio_mean = float(reused_clipped.mean().item())
        is_ratio_raw_max = float(reused_raw.max().item())
        is_clipped_frac = float((reused_raw > clip_max).float().mean().item())
    else:
        is_ratio_mean = 1.0
        is_ratio_raw_max = 1.0
        is_clipped_frac = 0.0

    metrics = {
        "training/overflow_buffer/reused_is_ratio_mean": is_ratio_mean,
        "training/overflow_buffer/reused_is_ratio_raw_max": is_ratio_raw_max,
        "training/overflow_buffer/reused_is_clipped_frac": is_clipped_frac,
    }
    return weights, metrics


class OverflowReuseReplayBuffer(ReplayBuffer):
    """V1 Sync DAPO ReplayBuffer with single-step overflow group reuse (6.4).

    Qualified groups left over after filling ``batch_size`` are kept in TransferQueue
    for at most ``max_reuse_steps`` (default 1) step. On the next step, buffered groups
    are prioritized ahead of newly generated groups unless the previous step's reuse
    ratio exceeded ``max_reuse_ratio`` (default 0.5), in which case reuse pauses for
    one step and buffered groups are cleared.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int | None,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig | dict[str, Any],
        poll_interval: float = 2.0,
        refill_fn=None,
        **kwargs: Any,
    ) -> None:
        cfg = dict(sampler_kwargs or {})
        filter_groups_metric = kwargs.pop("filter_groups_metric", cfg.get("filter_groups_metric", "reward"))
        train_batch_size = int(kwargs.pop("train_batch_size", cfg.get("train_batch_size", 64)))
        gen_batch_size = int(kwargs.pop("gen_batch_size", cfg.get("gen_batch_size", 1)))
        max_inflight_gen_batches = int(kwargs.pop("max_inflight_gen_batches", cfg.get("max_inflight_gen_batches", 2)))
        sync_refill_failed_groups = bool(
            kwargs.pop("sync_refill_failed_groups", cfg.get("sync_refill_failed_groups", False))
        )

        self.max_reuse_steps = int(cfg.get("max_reuse_steps", 1))
        self.max_reuse_ratio = float(cfg.get("max_reuse_ratio", 0.5))
        self.reuse_enabled = bool(cfg.get("reuse_enabled", True))
        self.is_ratio_clip = float(cfg.get("is_ratio_clip", 2.0))

        # partition_id -> {uid: source_global_step}
        self._overflow_buffer: dict[str, dict[str, int]] = {"train": {}}
        self._pause_next_reuse: dict[str, bool] = {"train": False}
        self._generated_prompts_this_step: dict[str, int] = {"train": 0}

        super().__init__(
            trainer_mode=trainer_mode,
            trainer_config=trainer_config,
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=sampler_kwargs,
            poll_interval=poll_interval,
            refill_fn=refill_fn,
            filter_groups_metric=filter_groups_metric,
            train_batch_size=train_batch_size,
            gen_batch_size=gen_batch_size,
            max_inflight_gen_batches=max_inflight_gen_batches,
            sync_refill_failed_groups=sync_refill_failed_groups,
        )

    def note_initial_submission(self, num_prompts: int, partition_id: str = "train") -> None:
        """Record prompts submitted in ``prepare_step`` for oversampling factor reporting."""
        self._generated_prompts_this_step[partition_id] = int(num_prompts)

    def get_reusable_count(self, global_steps: int, partition_id: str = "train") -> int:
        """Return how many valid buffered groups are eligible for reuse at ``global_steps``."""
        if not self.reuse_enabled or partition_id == "val":
            return 0
        if self._pause_next_reuse.get(partition_id, False):
            return 0
        buf = self._overflow_buffer.get(partition_id, {})
        valid = [uid for uid, src_step in buf.items() if 0 < (global_steps - src_step) <= self.max_reuse_steps]
        return min(len(valid), int(self.train_batch_size or len(valid)))

    def _evict_expired_overflow(self, global_steps: int, partition_id: str) -> tuple[int, bool]:
        """Clear buffered groups that are expired (> max_reuse_steps) or skipped due to pause."""
        if partition_id == "val":
            return 0, False
        buf = self._overflow_buffer.setdefault(partition_id, {})
        paused = self._pause_next_reuse.get(partition_id, False)
        if not buf:
            if paused:
                self._pause_next_reuse[partition_id] = False
            return 0, paused

        if not self.reuse_enabled or paused:
            expired_uids = set(buf.keys())
            buf.clear()
            self._pause_next_reuse[partition_id] = False
        else:
            expired_uids = {
                uid
                for uid, src_step in list(buf.items())
                if (global_steps - src_step) > self.max_reuse_steps or (global_steps - src_step) < 0
            }
            for uid in expired_uids:
                buf.pop(uid, None)

        if expired_uids:
            self._clear_groups(partition_id, expired_uids)
        return len(expired_uids), paused

    def _select_prompt_uids_with_buffer(
        self,
        partition_id: str,
        sampleable_keys: set[str],
        batch_size: int,
        buffered_uids: dict[str, int],
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        """Prioritize buffered overflow groups first, then oldest fresh prompt groups."""
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])

        buffered_candidates = sorted(
            [uid for uid in sampleable_keys if uid in buffered_uids],
            key=lambda uid: (buffered_uids[uid], uid),
        )
        fresh_candidates = sorted(
            [uid for uid in sampleable_keys if uid not in buffered_uids],
            key=lambda uid: (prompt_global_steps_snapshot.get(uid, 0), uid),
        )
        ordered = buffered_candidates + fresh_candidates
        return ordered[:batch_size], partition_snapshot, prompt_global_steps_snapshot

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch, prioritizing 1-step buffered overflow groups and saving new surplus."""
        if partition_id == "val" or not self.reuse_enabled:
            return super().sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)

        last_debug_time = time.time()
        eviction_metrics: dict[str, Any] = {}
        dapo_enabled = self.filter_groups_metric is not None
        refill_credit = 0
        draining = False
        max_inflight_prompts = self.max_inflight_gen_batches * (self.train_batch_size or batch_size)

        expired_cleared, reuse_paused = self._evict_expired_overflow(global_steps, partition_id)
        active_buffer = dict(self._overflow_buffer.get(partition_id, {}))
        buffer_size_in = len(active_buffer)

        while True:
            self._sync_metadata_from_transfer_queue()

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            failed_count = len(eviction_reasons[2])
            evicted_uids, stale_count, dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)
                for uid in evicted_uids:
                    active_buffer.pop(uid, None)

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            has_enough_samples = len(sampleable_keys) >= batch_size
            inflight_count = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])

            if not dapo_enabled and failed_count > 0 and not has_enough_samples:
                self._generated_prompts_this_step[partition_id] = (
                    self._generated_prompts_this_step.get(partition_id, 0) + failed_count
                )
                self.refill_fn(failed_count)
                continue

            if dapo_enabled:
                if has_enough_samples:
                    draining = True
                    refill_credit = 0
                elif not draining:
                    refill_credit += 2 * dapo_count + failed_count

                if not draining and refill_credit > 0:
                    available_slots = max(0, max_inflight_prompts - inflight_count)
                    dispatch_count = min(refill_credit, available_slots)
                    assert self.gen_batch_size is not None
                    dispatch_count -= dispatch_count % self.gen_batch_size
                    if dispatch_count > 0:
                        assert self.refill_fn is not None
                        self._generated_prompts_this_step[partition_id] = (
                            self._generated_prompts_this_step.get(partition_id, 0) + dispatch_count
                        )
                        self.refill_fn(dispatch_count)
                        refill_credit -= dispatch_count
                        continue

            can_select = has_enough_samples and (not dapo_enabled or inflight_count == 0)
            if can_select:
                selected_prompt_uids, partition_snapshot, _ = self._select_prompt_uids_with_buffer(
                    partition_id=partition_id,
                    sampleable_keys=sampleable_keys,
                    batch_size=batch_size,
                    buffered_uids=active_buffer,
                )
                selected_set = set(selected_prompt_uids)

                # Clear any unused previously-buffered groups (strictly 1-step TTL).
                unused_old_buffered = {uid for uid in active_buffer if uid not in selected_set}
                if unused_old_buffered:
                    self._clear_groups(partition_id, unused_old_buffered)
                    expired_cleared += len(unused_old_buffered)

                # Store newly generated qualified surplus groups for the next step only.
                surplus_new_uids = {
                    uid for uid in (sampleable_keys - selected_set) if uid not in active_buffer
                }
                self._overflow_buffer[partition_id] = {uid: global_steps for uid in surplus_new_uids}
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        # Clear prompt marker keys for selected UIDs and tag reused overflow trajectories.
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)
        selected_set = set(selected_prompt_uids)
        reused_selected_uids = selected_set & set(active_buffer.keys())

        keys, tags = [], []
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected_set:
                tag_copy = dict(tag)
                is_reused = uid in reused_selected_uids
                tag_copy["is_reused_overflow"] = is_reused
                tag_copy["overflow_age_steps"] = (global_steps - active_buffer[uid]) if is_reused else 0
                keys.append(key)
                tags.append(tag_copy)

        if not keys:
            raise RuntimeError("OverflowReuseReplayBuffer selected terminal groups with no materializable trajectories.")

        reused_groups = len(reused_selected_uids)
        reused_ratio = reused_groups / max(1, len(selected_prompt_uids))
        self._pause_next_reuse[partition_id] = reused_ratio > self.max_reuse_ratio

        gen_count = self._generated_prompts_this_step.pop(partition_id, 0)
        prefix = self._metrics_prefix(partition_id)
        eviction_metrics.update(
            {
                f"{prefix}/overflow_buffer/buffer_size_in": float(buffer_size_in),
                f"{prefix}/overflow_buffer/buffer_size_out": float(len(self._overflow_buffer[partition_id])),
                f"{prefix}/overflow_buffer/reused_groups": float(reused_groups),
                f"{prefix}/overflow_buffer/reused_ratio": float(reused_ratio),
                f"{prefix}/overflow_buffer/reuse_paused": 1.0 if reuse_paused else 0.0,
                f"{prefix}/overflow_buffer/expired_cleared": float(expired_cleared),
                f"{prefix}/overflow_buffer/oversample_factor": float(gen_count) / max(1, batch_size),
            }
        )
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags), eviction_metrics
