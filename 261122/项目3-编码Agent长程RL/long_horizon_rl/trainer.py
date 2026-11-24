"""基于 verl 0.9.1 V1 ``PPOTrainerSync`` 的编码长程 RL 训练器。

覆写要点（严格遵循通用约束）：
1. ``_compute_advantage``：
   - 沿用 ``compute_advantage_for_multi_trajectories`` 让同一轨迹全部段共享轨迹级 GRPO advantage；
   - 按 prompt ``uid`` 与规范化报错 anchor 组内计算 GiGPO step 级 advantage 并以 $\\omega=0.5$ 叠加；
   - 计算整条轨迹 token 总数归一权重 ``traj_token_weights``；
   - 通过 ``response_to_nested`` 与 ``tq.kv_batch_put`` 把 ``advantages``、``returns`` 与
     ``traj_token_weights`` 写回 TransferQueue 张量字段，确保传递到 actor loss。
2. ``prepare_step`` / ``on_step_end``：
   - 预读取下一批任务并在后台预热沙箱容器池；
   - 每 30 步执行阶段二题池全通过题目清理与高分轨迹（top-50）篡改抽检。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils.logging_utils import configure_verl_logging
from verl.workers.utils.padding import response_to_nested

from long_horizon_rl.agent_loop import CodingAgentLoopManager
from long_horizon_rl.sandbox_pool import SandboxPool
from long_horizon_rl.step_advantage import StepTokenSpan, build_combined_token_advantages
from long_horizon_rl.tamper_scan import append_tamper_log, scan_code_snippet

logger = logging.getLogger(__name__)


class CodingPPOTrainer(PPOTrainerSync):
    """集成多段 compaction advantage 共享、GiGPO step 级信用分配与容器池预热的同步 PPOTrainer。"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        coding_cfg = config.get("coding_rl", {})
        self.step_adv_cfg = coding_cfg.get("step_advantage", {})
        self.compaction_cfg = coding_cfg.get("compaction", {})
        self.reward_cfg = coding_cfg.get("reward", {})
        self.curriculum_cfg = coding_cfg.get("curriculum", {})
        self.sandbox_cfg = coding_cfg.get("sandbox", {})

        self.driver_pool: SandboxPool | None = None
        if bool(self.sandbox_cfg.get("enable", True)) and str(self.curriculum_cfg.get("stage", "repo")) == "repo":
            self.driver_pool = SandboxPool(
                pool_size_per_task=int(self.sandbox_cfg.get("pool_size_per_task", 10)),
                image_prefix=str(self.sandbox_cfg.get("image_prefix", "lhrl.task")),
                use_docker=bool(self.sandbox_cfg.get("use_docker", True)),
            )

    def _extract_task_records(self, batch: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if "extra_info" not in batch.keys():
            return records
        for item in batch["extra_info"]:
            info = getattr(item, "data", item)
            if isinstance(info, str):
                info = json.loads(info)
            if isinstance(info, dict) and info.get("task_payload_json"):
                payload = info["task_payload_json"]
                records.append(json.loads(payload) if isinstance(payload, str) else payload)
        return records

    def prepare_step(self) -> dict:
        """在提交当前步 rollout 的同时，对阶段二任务触发沙箱预热。"""
        next_batch = self._next_train_batch()
        if (
            bool(self.sandbox_cfg.get("enable", True))
            and str(self.curriculum_cfg.get("stage", "repo")) == "repo"
            and hasattr(self.agent_loop_manager, "prewarm_next_tasks")
        ):
            tasks = self._extract_task_records(next_batch)
            self.agent_loop_manager.prewarm_next_tasks(tasks)
        self._submit_batch_to_rollout(next_batch)
        return {}

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
            "extra_fields",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        raw_extra_fields = list(data.pop("extra_fields")) if "extra_fields" in data.keys() else [{}] * len(batch.keys)
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

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

        # 3.3 轨迹级 GRPO advantage：仅对每个 session 末段计算后广播到同轨迹全部分段
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 3.4 GiGPO step 级 advantage + 3.3 轨迹总 token 归一权重
        session_rewards: dict[str, float] = {}
        session_spans: dict[str, list[StepTokenSpan]] = {}
        tamper_count = 0
        compaction_count = 0
        seen_sessions: set[str] = set()

        row_scores = data.batch["rm_scores"].sum(dim=-1).tolist()
        for row_idx, key in enumerate(batch.keys):
            parts = key.rsplit("_", 2)
            session_key = f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
            session_rewards[session_key] = float(row_scores[row_idx])

            extra = raw_extra_fields[row_idx]
            extra_dict = getattr(extra, "data", extra) if extra is not None else {}
            if not isinstance(extra_dict, dict):
                extra_dict = {}

            if session_key not in seen_sessions:
                seen_sessions.add(session_key)
                reward_extra = extra_dict.get("reward_extra_info") or {}
                if reward_extra.get("tamper_flag"):
                    tamper_count += 1
                if extra_dict.get("compaction_triggered") or reward_extra.get("compaction_triggered"):
                    compaction_count += 1

            spans_raw = extra_dict.get("step_spans_json")
            if spans_raw:
                parsed = json.loads(spans_raw) if isinstance(spans_raw, str) else spans_raw
                for sp in parsed:
                    session_spans.setdefault(session_key, []).append(
                        StepTokenSpan(
                            segment_idx=int(sp["segment_idx"]),
                            start_token=int(sp["start_token"]),
                            end_token=int(sp["end_token"]),
                            anchor=sp.get("anchor"),
                            step_index=int(sp["step_index"]),
                        )
                    )

        step_enable = bool(self.step_adv_cfg.get("enable", True))
        omega = float(self.step_adv_cfg.get("omega", 0.5)) if step_enable else 0.0
        gamma = float(self.step_adv_cfg.get("gamma", 0.95))
        min_group_size = int(self.step_adv_cfg.get("min_group_size", 2))
        group_by_uid = bool(self.step_adv_cfg.get("group_by_uid", False))

        combined_adv, traj_weights, diag_metrics = build_combined_token_advantages(
            base_advantages=data.batch["advantages"],
            response_mask=data.batch["response_mask"],
            batch_keys=batch.keys,
            session_rewards_by_key=session_rewards,
            session_spans_by_session=session_spans,
            omega=omega,
            gamma=gamma,
            min_group_size=min_group_size,
            group_by_uid=group_by_uid,
        )
        data.batch["advantages"] = combined_adv
        if bool(self.compaction_cfg.get("normalize_by_trajectory_tokens", True)):
            data.batch["traj_token_weights"] = traj_weights

        num_sessions = max(1, len(seen_sessions))
        diag_metrics["tamper/flagged_ratio"] = float(tamper_count / num_sessions)
        diag_metrics["compaction/triggered_ratio"] = float(compaction_count / num_sessions)
        metrics.update(diag_metrics)

        write_fields = ["advantages", "returns"]
        if "traj_token_weights" in data.batch:
            write_fields.append("traj_token_weights")
        if self.config.algorithm.use_kl_in_reward:
            write_fields.append("token_level_rewards")
        if rollout_correction:
            write_fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                write_fields.append("rollout_is_weights")

        output = {field: response_to_nested(data.batch[field], response_mask) for field in write_fields}
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output, batch_size=len(batch)),
        )
        return batch

    def on_step_end(self):
        super().on_step_end()
        audit_interval = int(self.reward_cfg.get("audit_interval_steps", 30))
        if audit_interval > 0 and self.global_steps > 0 and self.global_steps % audit_interval == 0:
            self._audit_recent_high_reward_dump()

    def _audit_recent_high_reward_dump(self) -> None:
        """每 30 步对最近落盘的 rollout 高分样本做一次额外静态篡改复检。"""
        rollout_dir = self.config.trainer.get("rollout_data_dir")
        if not rollout_dir:
            return
        dump_file = Path(rollout_dir) / f"{self.global_steps}.jsonl"
        if not dump_file.is_file():
            return
        top_k = int(self.reward_cfg.get("audit_top_k", 50))
        records = []
        with dump_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        records.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        tamper_log = Path(str(self.reward_cfg.get("tamper_log_path", "logs/tamper_events.jsonl")))
        for item in records[:top_k]:
            res = scan_code_snippet(str(item.get("output", "")))
            if res.flagged:
                append_tamper_log(
                    tamper_log,
                    task_id=str(item.get("uid", "")),
                    step=self.global_steps,
                    scan_result=res,
                    trajectory_id="audit_top50",
                )

    def on_train_end(self):
        super().on_train_end()
        if self.driver_pool is not None:
            self.driver_pool.close()


@ray.remote
class CodingTrainingRunner:
    """V1 TrainingRunner（参照 verl ``TaskRunnerV1`` 结构）。"""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        self.agent_loop_manager = CodingAgentLoopManager.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()
        import transfer_queue as tq

        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = CodingPPOTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()
