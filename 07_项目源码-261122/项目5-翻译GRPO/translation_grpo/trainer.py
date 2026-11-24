"""Custom V1 PPOTrainerSync subclass for translation GRPO (5.1-5.4)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1 import PPOTrainerSync
from verl.workers.utils.padding import response_to_nested

from .advantage_mask import AnchorAdvantageConfig, compute_translation_grpo_advantage
from .judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig, JudgeOutputError
from .mqm_protocol import MQM_DIMENSIONS, score_group_with_anchor_hybrid
from .reward_scoring import (
    RewardWeightsConfig,
    compute_seven_dim_reward,
    detect_invalid_output,
    load_sidecar_config,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TranslationGRPOTrainer(PPOTrainerSync):
    """GRPO trainer overriding _compute_advantage for 7-dim MQM, anchor shaping, and hard masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_dim_history: dict[str, list[float]] = defaultdict(list)
        self._judge_client: AsyncOpenAIJudgeClient | None = None

    def _get_translation_cfg(self) -> dict[str, Any]:
        sidecar = load_sidecar_config()
        cfg = self.config.get("translation_grpo", None)
        if cfg is None:
            return sidecar
        return {**sidecar, **dict(cfg)}

    def _get_reward_weights_config(self) -> RewardWeightsConfig:
        t_cfg = self._get_translation_cfg()
        r_cfg = dict(t_cfg.get("reward", {}))
        return RewardWeightsConfig(
            **{k: v for k, v in r_cfg.items() if hasattr(RewardWeightsConfig, k)}
        )

    def _get_anchor_adv_config(self) -> AnchorAdvantageConfig:
        t_cfg = self._get_translation_cfg()
        a_cfg = dict(t_cfg.get("advantage", {}))
        return AnchorAdvantageConfig(
            anchor_enabled=bool(a_cfg.get("anchor_enabled", True)),
            anchor_lambda=float(a_cfg.get("anchor_lambda", 0.3)),
            hard_mask_enabled=bool(a_cfg.get("hard_mask_enabled", True)),
            min_valid_group_size=int(a_cfg.get("min_valid_group_size", 2)),
            norm_adv_by_std_in_grpo=bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True)),
        )

    def _get_judge_client(self) -> AsyncOpenAIJudgeClient:
        if self._judge_client is None:
            t_cfg = self._get_translation_cfg()
            j_cfg = dict(t_cfg.get("judge", {}))
            client_cfg = JudgeClientConfig(
                **{k: v for k, v in j_cfg.items() if hasattr(JudgeClientConfig, k)}
            )
            self._judge_client = AsyncOpenAIJudgeClient(client_cfg)
        return self._judge_client

    @staticmethod
    def _run_coro(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(coro)
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    def _refine_groups_with_hybrid_anchor(
        self,
        *,
        uids: list[Any],
        responses_text: list[str],
        reward_extra_infos: list[dict[str, Any]],
        extra_infos: list[dict[str, Any]],
        is_padding_mask: np.ndarray,
        rewards_arr: np.ndarray,
        anchor_arr: np.ndarray,
        reward_cfg: RewardWeightsConfig,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Optionally run group-level shuffled 8+1 anchor scoring & close-pair swapped Pairwise."""
        t_cfg = self._get_translation_cfg()
        hybrid_in_trainer = bool(t_cfg.get("hybrid_pairwise_in_trainer", False))
        diff_threshold = float(t_cfg.get("pairwise_diff_threshold", 2.0))
        tiebreak_weight = float(t_cfg.get("pairwise_tiebreak_weight", 0.25))

        if not hybrid_in_trainer:
            return rewards_arr, anchor_arr, {}

        client = self._get_judge_client()
        group_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if not is_padding_mask[idx]:
                group_indices[uid].append(idx)

        async def _score_all_groups():
            tasks = []
            task_uids = []
            for uid, idxs in group_indices.items():
                first_info = extra_infos[idxs[0]] if idxs[0] < len(extra_infos) else {}
                source = str(first_info.get("source", ""))
                direction = str(first_info.get("direction", "zh2en"))
                reference = first_info.get("reference")
                cands = [responses_text[i] for i in idxs]
                task_uids.append(uid)
                tasks.append(
                    score_group_with_anchor_hybrid(
                        client,
                        source,
                        cands,
                        direction=direction,
                        reference_anchor=str(reference) if reference else None,
                        enable_pairwise_tiebreak=True,
                        diff_threshold=diff_threshold,
                        pairwise_tiebreak_weight=tiebreak_weight,
                        seed=hash(str(uid)) & 0xFFFFFFFF,
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            scored = {}
            for uid, res in zip(task_uids, results, strict=True):
                if isinstance(res, JudgeOutputError):
                    # 该组 judge 输出不合格：保留 reward worker 的逐条结果
                    logger.warning("组 %s 的混合判分失败，沿用逐条 reward: %s", uid, res)
                    continue
                if isinstance(res, BaseException):
                    raise res
                scored[uid] = res
            return scored

        group_results = self._run_coro(_score_all_groups())
        total_pw_calls = 0
        for uid, res in group_results.items():
            total_pw_calls += res.pairwise_calls
            idxs = group_indices[uid]
            first_info = extra_infos[idxs[0]] if idxs[0] < len(extra_infos) else {}
            source = str(first_info.get("source", ""))
            source_units = first_info.get("source_len")

            anc_reward_val = float("nan")
            if res.anchor_result is not None:
                anc_flags = detect_invalid_output(
                    str(first_info.get("reference", "")),
                    source,
                    source_units=source_units,
                    config=reward_cfg,
                )
                anc_7dim = compute_seven_dim_reward(
                    res.anchor_result.dim_penalties,
                    rep_flag=anc_flags["rep_flag"],
                    len_ratio=anc_flags["len_ratio"],
                    source_units=anc_flags["source_units"],
                    config=reward_cfg,
                )
                anc_reward_val = float(anc_7dim["score"])

            for local_pos, batch_idx in enumerate(idxs):
                r_info = reward_extra_infos[batch_idx]
                cand_mqm = res.candidate_results[local_pos]
                cand_7dim = compute_seven_dim_reward(
                    cand_mqm.dim_penalties,
                    rep_flag=bool(r_info.get("rep_flag", False)),
                    len_ratio=float(r_info.get("len_ratio", 1.0)),
                    source_units=int(r_info.get("source_units", source_units or 10)),
                    config=reward_cfg,
                )
                rewards_arr[batch_idx] = float(cand_7dim["score"] + res.pairwise_tiebreak_deltas[local_pos])
                if not np.isnan(anc_reward_val):
                    anchor_arr[batch_idx] = anc_reward_val

        return rewards_arr, anchor_arr, {"judge/hybrid_pairwise_calls": float(total_pw_calls)}

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute 7-dim reward, reference-anchor advantage shaping, and hard mask into TQ."""
        tensor_fields = [
            "uid",
            "responses",
            "response_mask",
            "loss_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
        ]
        meta_fields = ["extra_fields", "extra_info"]
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=tensor_fields + meta_fields,
        )

        # Pop non-tensor/nested metadata BEFORE calling to_padded_tensor().
        raw_extra_fields = list(data.pop("extra_fields")) if "extra_fields" in data else [{} for _ in batch.keys]
        raw_extra_info = list(data.pop("extra_info")) if "extra_info" in data else [{} for _ in batch.keys]

        nested_response_mask = data["response_mask"]
        nested_responses = data["responses"]
        padded_responses = nested_responses.to_padded_tensor(padding=self.tokenizer.pad_token_id)
        response_lengths = nested_response_mask.offsets().diff()

        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        uids_list = data.batch.pop("uid").tolist()
        data.non_tensor_batch["uid"] = np.array(uids_list, dtype=object)

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        bsz = len(batch.keys)
        is_padding_arr = np.array([bool(tag.get("is_padding", False)) for tag in batch.tags], dtype=bool)
        reward_cfg = self._get_reward_weights_config()
        adv_cfg = self._get_anchor_adv_config()

        reward_extra_infos: list[dict[str, Any]] = []
        extra_infos: list[dict[str, Any]] = []
        for idx in range(bsz):
            ef = getattr(raw_extra_fields[idx], "data", raw_extra_fields[idx])
            ef_dict = ef if isinstance(ef, dict) else {}
            rei = ef_dict.get("reward_extra_info", {})
            reward_extra_infos.append(rei if isinstance(rei, dict) else {})

            ei = getattr(raw_extra_info[idx], "data", raw_extra_info[idx])
            extra_infos.append(ei if isinstance(ei, dict) else {})

        # Base scalar rewards from token_level_rewards (e.g. rm_scores written by reward worker).
        base_scalar_rewards = data.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy().astype(np.float32)
        anchor_arr = np.full(bsz, np.nan, dtype=np.float32)
        invalid_arr = np.zeros(bsz, dtype=bool)
        rep_arr = np.zeros(bsz, dtype=bool)
        len_ratio_arr = np.ones(bsz, dtype=np.float32)

        dim_metric_lists: dict[str, list[float]] = defaultdict(list)
        responses_text: list[str] = []

        for idx in range(bsz):
            if is_padding_arr[idx]:
                responses_text.append("")
                continue

            rei = reward_extra_infos[idx]
            ei = extra_infos[idx]
            resp_len_tok = int(response_lengths[idx].item())
            resp_str = self.tokenizer.decode(
                padded_responses[idx, :resp_len_tok],
                skip_special_tokens=True,
            )
            responses_text.append(resp_str)

            if "anchor_reward" in rei and rei["anchor_reward"] is not None:
                anchor_arr[idx] = float(rei["anchor_reward"])

            # Re-verify invalid conditions from both reward_extra_info and token lengths.
            source_text = str(ei.get("source", ""))
            source_units = ei.get("source_len")
            flags = detect_invalid_output(
                resp_str,
                source_text,
                source_units=source_units,
                response_token_len=resp_len_tok,
                config=reward_cfg,
            )
            is_inv = bool(rei.get("is_invalid", False) or flags["is_invalid"])
            invalid_arr[idx] = is_inv
            rep_arr[idx] = bool(rei.get("rep_flag", False) or flags["rep_flag"])
            len_ratio_arr[idx] = float(rei.get("len_ratio", flags["len_ratio"]))

            for dim in MQM_DIMENSIONS:
                key = f"{dim}_penalty"
                if key in rei and rei[key] is not None:
                    dim_metric_lists[key].append(float(rei[key]))
            for extra_key in ("mqm_penalty", "rep_penalty", "len_penalty"):
                if extra_key in rei and rei[extra_key] is not None:
                    dim_metric_lists[extra_key].append(float(rei[extra_key]))

        base_scalar_rewards, anchor_arr, hybrid_metrics = self._refine_groups_with_hybrid_anchor(
            uids=uids_list,
            responses_text=responses_text,
            reward_extra_infos=reward_extra_infos,
            extra_infos=extra_infos,
            is_padding_mask=is_padding_arr,
            rewards_arr=base_scalar_rewards,
            anchor_arr=anchor_arr,
            reward_cfg=reward_cfg,
        )
        metrics.update(hybrid_metrics)

        device = data.batch["response_mask"].device
        adv_out = compute_translation_grpo_advantage(
            rewards=torch.tensor(base_scalar_rewards, device=device, dtype=torch.float32),
            response_mask=data.batch["response_mask"],
            uids=uids_list,
            invalid_mask=torch.tensor(invalid_arr, device=device, dtype=torch.bool),
            anchor_rewards=torch.tensor(anchor_arr, device=device, dtype=torch.float32),
            is_padding_mask=torch.tensor(is_padding_arr, device=device, dtype=torch.bool),
            loss_mask=data.batch.get("loss_mask"),
            config=adv_cfg,
        )

        data.batch["advantages"] = adv_out["advantages"]
        data.batch["returns"] = adv_out["returns"]
        data.batch["response_mask"] = adv_out["response_mask"]
        data.batch["loss_mask"] = adv_out["loss_mask"]

        metrics.update(adv_out["metrics"])
        non_pad = ~is_padding_arr
        if non_pad.any():
            metrics["reward/repetition_rate_flag_mean"] = float(rep_arr[non_pad].mean())
            metrics["reward/len_ratio_mean"] = float(len_ratio_arr[non_pad].mean())
        for key, vals in dim_metric_lists.items():
            if vals:
                metrics[f"reward/{key}_mean"] = float(sum(vals) / len(vals))

        # 5.4 Periodic mask-isolation sanity check (every 50 steps): verify masked rows have zero adv & loss_mask.
        grad_check_freq = int(self._get_translation_cfg().get("grad_check_freq", 50))
        if grad_check_freq > 0 and self.global_steps % grad_check_freq == 0:
            inv_t = torch.tensor(invalid_arr & non_pad, device=device, dtype=torch.bool)
            if inv_t.any():
                masked_adv_max = float(data.batch["advantages"][inv_t].abs().max().item())
                masked_lmask_sum = float(data.batch["loss_mask"][inv_t].abs().sum().item())
            else:
                masked_adv_max = 0.0
                masked_lmask_sum = 0.0
            metrics["diag/hard_mask_invalid_adv_max"] = masked_adv_max
            metrics["diag/hard_mask_invalid_loss_mask_sum"] = masked_lmask_sum

        fields_to_write = ["advantages", "returns", "response_mask", "loss_mask"]
        if self.config.algorithm.use_kl_in_reward:
            fields_to_write.append("token_level_rewards")
        if rollout_correction and "rollout_is_weights" in data.batch:
            fields_to_write.append("rollout_is_weights")

        output = {}
        for field_name in fields_to_write:
            output[field_name] = response_to_nested(data.batch[field_name], nested_response_mask)
        output_td = TensorDict(output, batch_size=len(batch))
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output_td)
        return batch

    def _validate(self) -> dict[str, float]:
        """Run validation and alert if any MQM dimension penalty rises twice consecutively (5.2)."""
        val_metrics = super()._validate()
        for dim in MQM_DIMENSIONS:
            matching = [
                float(v)
                for k, v in val_metrics.items()
                if f"{dim}_penalty" in k and isinstance(v, int | float | np.number)
            ]
            if not matching:
                continue
            mean_val = float(sum(matching) / len(matching))
            hist = self._val_dim_history[dim]
            hist.append(mean_val)
            if len(hist) >= 3 and hist[-1] > hist[-2] > hist[-3]:
                logger.warning(
                    "[TranslationGRPO][5.2 Alert] Validation penalty for dimension '%s' increased twice "
                    "consecutively: %.4f -> %.4f -> %.4f",
                    dim,
                    hist[-3],
                    hist[-2],
                    hist[-1],
                )
                val_metrics[f"val-alert/{dim}_consecutive_rise"] = 1.0
            else:
                val_metrics[f"val-alert/{dim}_consecutive_rise"] = 0.0
        return val_metrics
