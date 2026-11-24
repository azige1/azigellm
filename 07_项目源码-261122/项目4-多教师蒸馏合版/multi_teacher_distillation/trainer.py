import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import response_to_nested

from . import DEFAULT_DOMAIN_TO_TEACHER, ROUTED_DOMAINS, SELF_ANCHOR_DOMAIN, DISTILL_TEACHER_DOMAINS
from .domain_advantage import DomainAdvantageNormalizer
from .fixed_sampling import FixedDomainSampler
from .losses import compute_domain_token_weights
from .reliability_weighting import (
    CalibrationManager,
    build_batch_reliability_weights,
    load_calibration_tables,
)
from .rollout_masking import is_truncated, summarize_rollouts
from .routing import (
    check_routing_share_drift,
    filter_cached_logprobs_by_version,
    validate_routing_stage,
)
from .teacher.pool_requests import TeacherPoolClient
from .teacher.schemas import ScoreRequest
from .teacher_merging import load_domain_to_teacher
from .termination import terminal_token_ids

logger = logging.getLogger(__name__)


def extract_domain_scores(metrics: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for key, value in metrics.items():
        parts = key.split("/")
        for domain in ROUTED_DOMAINS:
            if domain in parts and isinstance(value, (int, float)) and any(
                marker in key for marker in ("reward", "score", "acc", "success")
            ):
                scores[domain] = float(value) * (100.0 if float(value) <= 1.0 else 1.0)
    return scores


class DistillationTrainer(PPOTrainerSync):
    def __init__(self, config) -> None:
        self._enable_hybrid_replicas = bool(config.actor_rollout_ref.get("hybrid_engine", True))
        self.standalone_checkpoint_manager = None
        self._resume_training_state = None
        self.target_shares: dict[str, float] | None = None
        self._latest_domain_kl: dict[str, float] = {}
        self.domain_to_teacher: dict[str, str] = dict(DEFAULT_DOMAIN_TO_TEACHER)
        self.advantage_normalizer: DomainAdvantageNormalizer | None = None
        self.calibration_manager: CalibrationManager | None = None
        super().__init__(config)
        anchor_from_ref = bool(config.get("teacher_scoring", {}).get("anchor_from_reference", True))
        reliability_enabled = bool(config.get("reliability_weighting", {}).get("enabled", True))
        if anchor_from_ref or reliability_enabled:
            self.use_reference_policy = True
        self.domain_sampler: FixedDomainSampler | None = None
        self.teacher_pool_client: TeacherPoolClient | None = None

    def on_init_end(self) -> None:
        if self.standalone_checkpoint_manager is not None:
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
        else:
            super().on_init_end()

    def on_step_end(self) -> None:
        if self.standalone_checkpoint_manager is not None:
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
        else:
            super().on_step_end()

    def on_sample_end(self) -> None:
        if self.standalone_checkpoint_manager is not None:
            self.standalone_checkpoint_manager.sleep_replicas()
            if self.curr_step_profile:
                self._stop_rollout_profiling()
        else:
            super().on_sample_end()

    def _get_n_gpus_for_throughput(self) -> int:
        total = super()._get_n_gpus_for_throughput()
        if self.standalone_checkpoint_manager is not None:
            rollout = self.config.actor_rollout_ref.rollout
            total += int(rollout.nnodes) * int(rollout.n_gpus_per_node)
        return total

    def _init_resource_pool_mgr(self) -> None:
        enabled = bool(self.config.distillation.enabled)
        self.config.distillation.enabled = False
        try:
            super()._init_resource_pool_mgr()
        finally:
            self.config.distillation.enabled = enabled

    def _setup(self) -> None:
        external = self.use_teacher_policy
        self.use_teacher_policy = False
        super()._setup()
        self.use_teacher_policy = external
        self.teacher_model_manager = None
        self.distillation_config = omega_conf_to_dataclass(self.config.distillation) if external else None

        if not self._enable_hybrid_replicas:
            self.llm_server_manager = LLMServerManager.create(config=self.config, start_rank=0)
            checkpoint_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
            self.standalone_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_config,
                actor_wg=self.actor_rollout_wg,
                replicas=self.llm_server_manager.get_replicas(),
            )

        routing_cfg = self.config.get("routing_validation", {}) or {}
        mapping_path = routing_cfg.get("domain_to_teacher_path")
        self.domain_to_teacher = load_domain_to_teacher(
            Path(str(mapping_path)) if mapping_path and Path(str(mapping_path)).exists() else None
        )

        adv_cfg = self.config.get("domain_advantage", {}) or {}
        self.advantage_normalizer = DomainAdvantageNormalizer(
            decay=float(adv_cfg.get("ema_decay", 0.9)),
            sigma_floor=float(adv_cfg.get("sigma_floor", 0.2)),
            raw_clip=float(adv_cfg.get("raw_clip", 5.0)),
            post_norm_clip=float(adv_cfg.get("post_norm_clip", 3.0)),
        )

        rel_cfg = self.config.get("reliability_weighting", {}) or {}
        tables = {}
        tables_path = rel_cfg.get("tables_path")
        if tables_path and Path(str(tables_path)).exists():
            tables = load_calibration_tables(Path(str(tables_path)))
        self.calibration_manager = CalibrationManager(
            tables=tables,
            refresh_interval=int(rel_cfg.get("refresh_interval", 40)),
            num_bins=int(rel_cfg.get("num_bins", 10)),
            alpha=float(rel_cfg.get("alpha", 2.0)),
            lift_cap=float(rel_cfg.get("lift_cap", 1.5)),
            lb_scale=float(rel_cfg.get("lb_scale", 2.0)),
        )

        if external:
            scoring = self.config.teacher_scoring
            versions = {name: str(item.model_version) for name, item in scoring.teachers.items()}
            self.teacher_pool_client = TeacherPoolClient(
                endpoints=list(scoring.endpoints),
                teacher_versions=versions,
                timeout=float(scoring.get("timeout", 1800)),
                max_attempts=int(scoring.get("max_attempts", 4)),
            )

    def get_teacher_client(self) -> None:
        return None

    def _init_dataloader(self) -> None:
        super()._init_dataloader()
        if self.config.distillation.enabled:
            self._build_domain_sampler()

    def _build_domain_sampler(self) -> None:
        rows = self.train_dataset.dataframe.to_dict("records")
        shares_path = self.config.domain_sampling.get("shares_path", None)
        if shares_path and Path(str(shares_path)).exists():
            shares = json.loads(Path(str(shares_path)).read_text(encoding="utf-8"))["shares"]
        else:
            shares = OmegaConf.to_container(self.config.domain_sampling.shares, resolve=True)
        domain_indices = {name: [] for name in shares}
        for index, row in enumerate(rows):
            domain = str(row.get("sampling_domain", row.get("domain", row.get("data_source"))))
            if domain not in domain_indices:
                continue
            domain_indices[domain].append(index)
        self.domain_sampler = FixedDomainSampler(
            domain_indices=domain_indices,
            shares={name: float(value) for name, value in shares.items()},
            seed=int(self.config.domain_sampling.get("seed", 17)),
        )
        self.target_shares = dict(self.domain_sampler.quota.shares)

    def _next_train_batch(self, num_prompts: int | None = None):
        if self.domain_sampler is None:
            return super()._next_train_batch(num_prompts)
        count = num_prompts or int(self.config.data.train_batch_size)
        indices = self.domain_sampler.sample_indices(count)
        batch_dict = collate_fn([self.train_dataset[index] for index in indices])
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    @staticmethod
    def _teacher_logprobs_from_response(response, response_ids: list[int], prompt_length: int) -> np.ndarray:
        length = len(response_ids)
        values = np.full(length, np.nan, dtype=np.float32)
        for item in response.scores:
            response_index = item.position - prompt_length
            if 0 <= response_index < length:
                if int(item.sampled_id) != int(response_ids[response_index]):
                    raise RuntimeError(
                        f"教师响应 token 对齐失败: pos={response_index}, "
                        f"teacher={item.sampled_id}, student={response_ids[response_index]}"
                    )
                values[response_index] = float(item.sampled_logprob)
        return values

    def _row_routing_dict(self, extra_field: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "domain": extra_field.get("domain", extra_info.get("domain")),
            "teacher": extra_field.get("teacher", extra_info.get("teacher")),
            "teacher_version": extra_field.get("teacher_version", extra_info.get("teacher_version", "")),
            "segment_index": extra_field.get("segment_index", extra_info.get("segment_index", 0)),
            "cached_teacher_version": extra_field.get("cached_teacher_version"),
            "cached_teacher_logprobs": extra_field.get("cached_teacher_logprobs"),
        }

    def _prepare_distillation(self, batch: KVBatchMeta, metrics: dict[str, Any]) -> KVBatchMeta:
        if self.teacher_pool_client is None:
            raise RuntimeError("教师客户端尚未初始化")
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "extra_fields",
            "extra_info",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        extra_fields = data.pop("extra_fields").tolist()
        extra_infos = data.pop("extra_info").tolist()
        prompt_lengths = data["prompts"].offsets().diff().tolist()
        response_lengths = data["responses"].offsets().diff().tolist()
        pad_id = self.tokenizer.pad_token_id
        prompts = data["prompts"].to_padded_tensor(padding=pad_id)
        responses = data["responses"].to_padded_tensor(padding=pad_id)
        nested_response_mask = data["response_mask"]
        masks = nested_response_mask.to_padded_tensor(padding=0).cpu().numpy()
        rollout = data["rollout_log_probs"].to_padded_tensor(padding=0.0).cpu().numpy()
        old = data["old_log_probs"].to_padded_tensor(padding=0.0).cpu().numpy()
        reference = data["ref_log_prob"].to_padded_tensor(padding=0.0).cpu().numpy()

        raw_advantages = torch.zeros(masks.shape, dtype=torch.float32)
        padded_masks = torch.zeros(masks.shape, dtype=torch.float32)
        padded_teacher_lps = torch.zeros(masks.shape, dtype=torch.float32)
        padded_ref_lps = torch.from_numpy(reference.astype(np.float32))
        active = [not tag.get("is_padding", False) for tag in batch.tags]
        limit = int(self.config.actor_rollout_ref.rollout.response_length)
        terminals = terminal_token_ids(self.tokenizer)

        mask_cfg = self.config.get("rollout_masking", {}) or {}
        mask_truncated = bool(mask_cfg.get("mask_truncated_rollouts", True))
        truncation_warning_threshold = float(mask_cfg.get("truncation_warning_threshold", 0.05))

        ratio_cfg = self.config.get("rollout_ratio_mask", {}) or {}
        ratio_mask_enabled = bool(ratio_cfg.get("enabled", True))
        min_ratio = float(ratio_cfg.get("min_ratio", 0.5))
        max_ratio = float(ratio_cfg.get("max_ratio", 2.0))

        routing_cfg = self.config.get("routing_validation", {}) or {}
        routing_strict_mode = str(routing_cfg.get("strict_mode", "raise"))
        drift_threshold = float(routing_cfg.get("share_drift_alarm_threshold", 0.01))

        adv_cfg = self.config.get("domain_advantage", {}) or {}
        adv_norm_enabled = bool(adv_cfg.get("enabled", True))
        raw_clip = float(adv_cfg.get("raw_clip", 5.0))
        post_norm_clip = float(adv_cfg.get("post_norm_clip", 3.0))

        rel_cfg = self.config.get("reliability_weighting", {}) or {}
        rel_enabled = bool(rel_cfg.get("enabled", True))
        rel_norm_within_domain = bool(rel_cfg.get("normalize_within_domain", True))

        reweight_cfg = self.config.get("domain_reweighting", {}) or {}
        reweight_enabled = bool(reweight_cfg.get("domain_token_reweight", True))

        # 4.3 阶段一：rollout 输出级路由校验
        active_indices = [idx for idx, flag in enumerate(active) if flag]
        active_routing_rows = [self._row_routing_dict(extra_fields[idx], extra_infos[idx]) for idx in active_indices]
        rollout_audit = validate_routing_stage(
            "rollout_output",
            active_routing_rows,
            domain_to_teacher=self.domain_to_teacher,
            strict_mode=routing_strict_mode,
        )
        metrics["distillation/routing_dropped/rollout_output"] = float(rollout_audit["dropped_count"])
        valid_active_set = {active_indices[i] for i in rollout_audit["valid_indices"]}

        # 版本失效检查：教师表变化时丢弃旧缓存重算
        version_audit = filter_cached_logprobs_by_version(
            [active_routing_rows[i] for i in rollout_audit["valid_indices"]],
            self.teacher_pool_client.teacher_versions,
        )
        metrics["distillation/teacher_logprob_recomputed"] = float(version_audit["recomputed"])

        requests: list[ScoreRequest] = []
        request_row_meta: list[dict[str, Any]] = []
        rollout_rows: list[dict[str, Any]] = []
        truncated_rows: list[bool] = [False] * len(batch.keys)
        domains: list[str] = []
        teachers: list[str] = []
        routed_token_counts: dict[str, int] = {}

        for row, key in enumerate(batch.keys):
            is_valid_row = row in valid_active_set
            if is_valid_row:
                routing = self._row_routing_dict(extra_fields[row], extra_infos[row])
                domain = str(routing["domain"])
                teacher = str(routing["teacher"])
            else:
                domain = SELF_ANCHOR_DOMAIN
                teacher = SELF_ANCHOR_DOMAIN
            domains.append(domain)
            teachers.append(teacher)

            response_ids = responses[row, : response_lengths[row]].tolist()
            truncated = active[row] and is_truncated(response_ids, limit, terminals)
            truncated_rows[row] = truncated
            extra_fields[row]["truncated"] = truncated

            if active[row] and is_valid_row:
                rollout_rows.append({"domain": domain, "response_ids": response_ids, "truncated": truncated})
            if not is_valid_row or (mask_truncated and truncated):
                continue

            token_count = int(masks[row, : response_lengths[row]].sum())
            routed_token_counts[domain] = routed_token_counts.get(domain, 0) + token_count

            if teacher == SELF_ANCHOR_DOMAIN and teacher not in self.teacher_pool_client.teacher_versions:
                continue

            positions = list(range(prompt_lengths[row], prompt_lengths[row] + response_lengths[row]))
            request_row_meta.append({
                "domain": domain,
                "teacher": teacher,
                "teacher_version": self.teacher_pool_client.teacher_versions.get(teacher, ""),
                "segment_index": extra_fields[row].get("segment_index", extra_infos[row].get("segment_index", 0)),
            })
            requests.append(
                ScoreRequest(
                    request_id=f"{key}::{teacher}",
                    teacher_id=teacher,
                    token_ids=prompts[row, : prompt_lengths[row]].tolist() + response_ids,
                    positions=positions,
                    top_k=0,
                    expected_version=self.teacher_pool_client.teacher_versions.get(teacher),
                )
            )

        # 4.3 阶段二：教师请求级路由校验
        if request_row_meta:
            req_audit = validate_routing_stage(
                "teacher_request",
                request_row_meta,
                domain_to_teacher=self.domain_to_teacher,
                expected_teacher_versions=self.teacher_pool_client.teacher_versions,
                strict_mode=routing_strict_mode,
                require_teacher_version=True,
            )
            metrics["distillation/routing_dropped/teacher_request"] = float(req_audit["dropped_count"])

        # 4.3 每步各领域路由 token 份额告警
        if self.target_shares and routed_token_counts:
            drift_report = check_routing_share_drift(
                routed_token_counts, self.target_shares, drift_threshold=drift_threshold
            )
            for d_name, drift_val in drift_report["drifts"].items():
                metrics[f"distillation/routing_share_drift/{d_name}"] = float(drift_val)
            metrics["distillation/routing_share_alarm"] = 1.0 if drift_report["alarm"] else 0.0
            if drift_report["alarm"]:
                logger.warning("路由 token 份额偏差超过 %.2f%%: %s", drift_threshold * 100, drift_report["alarm_domains"])

        scored = asyncio.run(self.teacher_pool_client.score_batch(requests)) if requests else []
        score_by_id = {response.request_id: response for response in scored}

        pre_loss_rows: list[dict[str, Any]] = []
        pre_loss_indices: list[int] = []

        for row, key in enumerate(batch.keys):
            if row not in valid_active_set or (mask_truncated and truncated_rows[row]):
                continue
            length = response_lengths[row]
            response_ids = responses[row, :length].tolist()
            domain = domains[row]
            teacher = teachers[row]

            if teacher == SELF_ANCHOR_DOMAIN and teacher not in self.teacher_pool_client.teacher_versions:
                teacher_lps = reference[row, :length].astype(np.float32)
                teacher_version = str(self.config.actor_rollout_ref.model.path)
                prefill_tokens = int(prompt_lengths[row] + length)
            else:
                response = score_by_id[f"{key}::{teacher}"]
                teacher_lps = self._teacher_logprobs_from_response(response, response_ids, prompt_lengths[row])
                teacher_version = response.model_version
                prefill_tokens = int(response.prefill_tokens)

            student = old[row, :length]
            if not np.isfinite(student).all():
                raise RuntimeError("old-log-prob 未与响应序列完整对齐")
            if not np.isfinite(teacher_lps).all():
                raise RuntimeError("教师 log-prob 未与响应序列完整对齐")

            original_mask = masks[row, :length].astype(bool)
            if ratio_mask_enabled:
                ratio = np.exp(np.clip(student - rollout[row, :length], -20, 20))
                ratio_mask = (ratio >= min_ratio) & (ratio <= max_ratio)
                valid = original_mask & ratio_mask
            else:
                ratio_mask = np.ones_like(original_mask, dtype=bool)
                valid = original_mask

            raw_diff = teacher_lps - student
            raw_advantages[row, :length] = torch.from_numpy((raw_diff * valid).astype(np.float32))
            padded_masks[row, :length] = torch.from_numpy(valid.astype(np.float32))
            padded_teacher_lps[row, :length] = torch.from_numpy(teacher_lps.astype(np.float32))

            extra_fields[row].update({
                "domain": domain,
                "teacher": teacher,
                "teacher_version": teacher_version,
                "prefill_tokens": prefill_tokens,
                "ratio_masked_tokens": int((original_mask & ~ratio_mask).sum()),
            })
            pre_loss_rows.append(self._row_routing_dict(extra_fields[row], extra_infos[row]))
            pre_loss_indices.append(row)

            denominator = max(int(original_mask.sum()), 1)
            metrics.setdefault(f"distillation/token_kl/{domain}", []).append(
                float((student - teacher_lps)[original_mask].sum() / denominator)
            )
            metrics.setdefault(f"distillation/ratio_masked_fraction/{domain}", []).append(
                float((original_mask & ~ratio_mask).sum() / denominator)
            )

        # 4.3 阶段三：loss 计算前最终路由校验
        if pre_loss_rows:
            pre_loss_audit = validate_routing_stage(
                "pre_loss",
                pre_loss_rows,
                domain_to_teacher=self.domain_to_teacher,
                strict_mode=routing_strict_mode,
            )
            metrics["distillation/routing_dropped/pre_loss"] = float(pre_loss_audit["dropped_count"])
            for dropped_pos in pre_loss_audit["dropped_indices"]:
                bad_row = pre_loss_indices[dropped_pos]
                padded_masks[bad_row].zero_()
                raw_advantages[bad_row].zero_()

        # 4.4 教师校准位移的 token 级可靠性加权
        if rel_enabled and self.calibration_manager is not None and self.calibration_manager.tables:
            padded_token_weights, rel_means = build_batch_reliability_weights(
                domains=domains,
                teacher_logprobs=padded_teacher_lps,
                ref_logprobs=padded_ref_lps,
                response_masks=padded_masks,
                tables=self.calibration_manager.tables,
                normalize_within_domain=rel_norm_within_domain,
            )
            for d_name, mean_val in rel_means.items():
                metrics[f"distillation/reliability_multiplier/{d_name}"] = float(mean_val)
        else:
            padded_token_weights = (padded_masks > 0).float()

        # 4.5 领域滑动标准差归一的 token 级 advantage
        if adv_norm_enabled and self.advantage_normalizer is not None:
            normalized_advantages, norm_diag = self.advantage_normalizer.normalize_batch(
                raw_advantages=raw_advantages,
                response_mask=padded_masks,
                domains=domains,
                clip_after_norm=True,
            )
            for d_name, s_val in norm_diag["step_stds"].items():
                metrics[f"distillation/step_std/{d_name}"] = float(s_val)
            for d_name, sig_val in norm_diag["effective_sigmas"].items():
                metrics[f"distillation/domain_sigma/{d_name}"] = float(sig_val)
        else:
            normalized_advantages = torch.clamp(raw_advantages, min=-raw_clip, max=raw_clip) * (padded_masks > 0).float()
            normalized_advantages = torch.clamp(normalized_advantages, min=-post_norm_clip, max=post_norm_clip)

        # 按实际梯度 token 量对各领域重加权，使各域梯度份额等于静态目标份额
        sample_weights = torch.ones(len(domains), dtype=torch.float32)
        if self.target_shares:
            effective_for_mass = normalized_advantages * padded_token_weights
            sample_weights, diagnostics = compute_domain_token_weights(
                effective_for_mass, padded_masks, domains, self.target_shares
            )
            for domain, value in diagnostics["actual_shares"].items():
                metrics[f"distillation/gradient_token_share/{domain}"] = float(value)
            for domain, value in diagnostics["target_shares"].items():
                metrics[f"distillation/target_share/{domain}"] = float(value)
            for domain, value in diagnostics["share_gaps"].items():
                metrics[f"distillation/gradient_share_gap/{domain}"] = float(value)
            for domain, value in diagnostics["domain_weights"].items():
                metrics[f"distillation/domain_weight/{domain}"] = float(value)
            if not reweight_enabled:
                sample_weights = torch.ones(len(domains), dtype=torch.float32)

        rollout_summary = (
            summarize_rollouts(rollout_rows, warning_threshold=truncation_warning_threshold)
            if rollout_rows
            else {"domains": {}}
        )
        for domain, values in rollout_summary["domains"].items():
            metrics[f"rollout/truncation_rate/{domain}"] = values["truncation_rate"]
            metrics[f"rollout/mean_length/{domain}"] = values["mean_length"]
            metrics[f"rollout/repeated_ngram_fraction/{domain}"] = values["repeated_ngram_fraction"]
            if values["warning"]:
                logger.warning("领域 %s 的截断率 %.3f 超过阈值", domain, values["truncation_rate"])

        for name, values in list(metrics.items()):
            if isinstance(values, list) and name.startswith("distillation/"):
                mean_val = float(np.mean(values))
                metrics[name] = mean_val
                if name.startswith("distillation/token_kl/"):
                    domain_name = name.rsplit("/", 1)[-1]
                    self._latest_domain_kl[domain_name] = mean_val

        padded_domain_weights = sample_weights.unsqueeze(-1).expand_as(padded_masks) * (padded_masks > 0).float()

        extra_array = np.empty(len(extra_fields), dtype=object)
        extra_array[:] = extra_fields
        output = tu.get_tensordict(
            {
                "distillation_advantages": response_to_nested(normalized_advantages, nested_response_mask),
                "distillation_token_weights": response_to_nested(padded_token_weights, nested_response_mask),
                "distillation_domain_weights": response_to_nested(padded_domain_weights, nested_response_mask),
                "response_mask": response_to_nested(padded_masks, nested_response_mask),
            },
            non_tensor_dict={"extra_fields": extra_array},
        )
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
        batch.extra_info["distillation_loss_coef"] = float(
            self.config.distillation.distillation_loss.distillation_loss_coef
        )
        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        if self.config.distillation.enabled:
            batch = self._prepare_distillation(batch, metrics)
        return super()._update_actor(batch, metrics)

    def _validate(self) -> dict[str, float]:
        metrics = super()._validate()
        log_path = self.config.get("evaluation_log", None)
        scores = extract_domain_scores(metrics)
        if log_path and scores:
            row: dict[str, Any] = {
                "step": self.global_steps,
                "checkpoint": str(self.config.actor_rollout_ref.model.path)
                if self.global_steps == 0
                else os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "huggingface"),
                "scores": {name: scores[name] for name in DISTILL_TEACHER_DOMAINS if name in scores},
            }
            if SELF_ANCHOR_DOMAIN in scores:
                row["self_anchor_score"] = scores[SELF_ANCHOR_DOMAIN]
            if self._latest_domain_kl:
                row["token_kl"] = dict(self._latest_domain_kl)
            path = Path(str(log_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return metrics

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "training_state.pt")
        torch.save(
            {
                "domain_sampler": self.domain_sampler.state_dict() if self.domain_sampler else None,
                "target_shares": dict(self.target_shares) if self.target_shares is not None else None,
                "advantage_normalizer": self.advantage_normalizer.state_dict() if self.advantage_normalizer else None,
                "calibration_manager": self.calibration_manager.state_dict() if self.calibration_manager else None,
            },
            path,
        )

    def _load_checkpoint(self) -> None:
        super()._load_checkpoint()
        if self.global_steps <= 0:
            return
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "training_state.pt")
        if not os.path.exists(path):
            raise RuntimeError("蒸馏 checkpoint 缺少 training_state.pt")
        state = torch.load(path, weights_only=False)
        self._resume_training_state = state
        if self.domain_sampler and state.get("domain_sampler"):
            self.domain_sampler.load_state_dict(state["domain_sampler"])
        if state.get("target_shares"):
            self.target_shares = dict(state["target_shares"])
        if self.advantage_normalizer and state.get("advantage_normalizer"):
            self.advantage_normalizer.load_state_dict(state["advantage_normalizer"])
        if self.calibration_manager and state.get("calibration_manager"):
            self.calibration_manager.load_state_dict(state["calibration_manager"])
