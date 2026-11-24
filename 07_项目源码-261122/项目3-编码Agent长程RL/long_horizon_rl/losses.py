"""自定义 PPO loss：支持多段轨迹按总 token 数归一权重（3.3 节）。

通过 Ray setup hook 替换 ``verl.workers.utils.losses.ppo_loss`` 与
``verl.workers.engine_workers.ppo_loss``，确保分段轨迹权重与标准 PPO/GRPO
token-mean 聚合严格对齐。
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


def coding_ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """在 verl 0.9.1 ``ppo_loss`` 基础上叠加 ``traj_token_weights``。

    当 ``traj_token_weights`` 存在时，同条轨迹全部段中的每个 assistant token（含摘要）
    以相同权重进入 policy loss 与 KL loss，使多段轨迹总权重与单段轨迹一致。
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if "traj_token_weights" in data:
        fields.append("traj_token_weights")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    traj_token_weights = data.get("traj_token_weights", None)

    if traj_token_weights is not None:
        effective_weights = (
            rollout_is_weights * traj_token_weights if rollout_is_weights is not None else traj_token_weights
        )
    else:
        effective_weights = rollout_is_weights

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=effective_weights,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None and config.entropy_coeff != 0.0:
        entropy_mat = entropy * traj_token_weights if traj_token_weights is not None else entropy
        entropy_loss = agg_loss(
            loss_mat=entropy_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        if traj_token_weights is not None:
            kld = kld * traj_token_weights
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )
        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
