"""V1 PPOTrainerSync subclass integrating pass-rate peak advantage (6.3) and overflow reuse IS (6.4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from math_reasoning_rl.config import OverflowReuseConfig, PassRateWeightConfig
from math_reasoning_rl.difficulty_advantage import compute_pass_rate_weighted_advantage
from math_reasoning_rl.overflow_replay_buffer import (
    OverflowReuseReplayBuffer,
    compute_truncated_importance_weights,
)
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_to_nested


class MathReasoningPPOTrainer(PPOTrainerSync):
    """Synchronous V1 DAPO trainer with pass-rate weighted advantages and single-step overflow reuse."""

    def __init__(self, config):
        self._pass_rate_cfg = self._load_pass_rate_config(config)
        self._overflow_cfg = self._load_overflow_config(config)
        super().__init__(config)

    @staticmethod
    def _load_pass_rate_config(config) -> PassRateWeightConfig:
        math_rl_cfg = config.get("math_rl", {}) or {}
        adv_cfg = math_rl_cfg.get("pass_rate_weight", {}) or {}
        return PassRateWeightConfig(
            center=float(adv_cfg.get("center", 0.25)),
            sigma=float(adv_cfg.get("sigma", 0.15)),
            amplitude=float(adv_cfg.get("amplitude", 0.6)),
            positive_reward_threshold=float(adv_cfg.get("positive_reward_threshold", 0.0)),
            enabled=bool(adv_cfg.get("enabled", True)),
        )

    @staticmethod
    def _load_overflow_config(config) -> OverflowReuseConfig:
        math_rl_cfg = config.get("math_rl", {}) or {}
        ov_cfg = math_rl_cfg.get("overflow_reuse", {}) or {}
        return OverflowReuseConfig(
            max_reuse_steps=int(ov_cfg.get("max_reuse_steps", 1)),
            max_reuse_ratio=float(ov_cfg.get("max_reuse_ratio", 0.5)),
            is_ratio_clip=float(ov_cfg.get("is_ratio_clip", 2.0)),
            enabled=bool(ov_cfg.get("enabled", True)),
        )

    def prepare_step(self) -> dict:
        """Subtract eligible buffered overflow groups from initial rollout submission."""
        train_batch_size = int(self.config.data.train_batch_size)
        reusable = 0
        if isinstance(self.replay_buffer, OverflowReuseReplayBuffer):
            reusable = self.replay_buffer.get_reusable_count(self.global_steps, partition_id="train")
        to_submit = max(0, train_batch_size - reusable)
        if isinstance(self.replay_buffer, OverflowReuseReplayBuffer):
            self.replay_buffer.note_initial_submission(to_submit, partition_id="train")
        if to_submit > 0:
            self._add_prompts_to_generate(to_submit)
        return {}

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict[str, Any]) -> KVBatchMeta:
        """Compute uid-grouped pass-rate weighted advantages (6.3) and truncated IS weights (6.4)."""
        fields = ["uid", "response_mask", "rm_scores", "old_log_probs"]
        if self._overflow_cfg.enabled and self.config.actor_rollout_ref.rollout.get("calculate_log_probs", True):
            fields.append("rollout_log_probs")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        nested_response_mask = data["response_mask"]
        padded = data.to_padded_tensor()
        response_mask = padded["response_mask"]
        token_level_rewards = padded["rm_scores"]
        uids = np.array(padded["uid"].tolist(), dtype=object)

        advantages, returns, adv_metrics = compute_pass_rate_weighted_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=uids,
            config=self._pass_rate_cfg,
        )
        metrics.update(adv_metrics)

        is_reused_flags = torch.tensor(
            [bool(tag.get("is_reused_overflow", False)) for tag in batch.tags],
            dtype=torch.bool,
        )
        write_fields: dict[str, torch.Tensor] = {
            "advantages": response_to_nested(advantages, nested_response_mask),
            "returns": response_to_nested(returns, nested_response_mask),
            "token_level_rewards": response_to_nested(token_level_rewards, nested_response_mask),
        }

        if self._overflow_cfg.enabled and "old_log_probs" in padded and "rollout_log_probs" in padded:
            is_weights, is_metrics = compute_truncated_importance_weights(
                old_log_probs=padded["old_log_probs"],
                sampling_log_probs=padded["rollout_log_probs"],
                response_mask=response_mask,
                is_reused_mask=is_reused_flags,
                clip_max=self._overflow_cfg.is_ratio_clip,
            )
            metrics.update(is_metrics)
            write_fields["rollout_is_weights"] = response_to_nested(is_weights, nested_response_mask)

        output = TensorDict(write_fields, batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
