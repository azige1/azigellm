from typing import Any, Mapping, Sequence

import torch
from tensordict import TensorDict
from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.workers.config import ActorConfig, DistillationConfig


def compute_domain_token_weights(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    domains: Sequence[str],
    target_shares: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    """按各领域实际梯度 token 量重加权，使各领域有效梯度份额等于目标份额。"""
    if advantages.shape != response_mask.shape:
        raise ValueError("advantage 必须与 response_mask 等形")
    if len(domains) != advantages.shape[0]:
        raise ValueError("样本领域标签数量必须与 batch 大小一致")
    valid = response_mask.bool()
    abs_adv = (advantages.detach().float().abs() * valid.float()).sum(dim=-1)

    domain_mass: dict[str, float] = {name: 0.0 for name in target_shares}
    for row_index, domain in enumerate(domains):
        key = str(domain)
        domain_mass[key] = domain_mass.get(key, 0.0) + float(abs_adv[row_index].item())

    total_mass = sum(domain_mass.values())
    actual_shares = {
        name: (mass / total_mass if total_mass > 0.0 else 0.0)
        for name, mass in domain_mass.items()
    }
    present_domains = [name for name, mass in domain_mass.items() if mass > 0.0 and name in target_shares]
    present_target_sum = sum(float(target_shares[name]) for name in present_domains)

    domain_weights: dict[str, float] = {}
    for name in set(target_shares) | set(domain_mass):
        mass = domain_mass.get(name, 0.0)
        if name in target_shares and mass > 0.0 and total_mass > 0.0 and present_target_sum > 0.0:
            effective_target = float(target_shares[name]) / present_target_sum
            domain_weights[name] = effective_target / actual_shares[name]
        else:
            domain_weights[name] = 1.0

    sample_weights = torch.tensor(
        [domain_weights.get(str(domain), 1.0) for domain in domains],
        device=advantages.device,
        dtype=torch.float32,
    )
    share_gaps = {
        name: actual_shares.get(name, 0.0) - float(target_shares.get(name, 0.0))
        for name in target_shares
    }
    diagnostics = {
        "actual_shares": actual_shares,
        "target_shares": {name: float(value) for name, value in target_shares.items()},
        "share_gaps": share_gaps,
        "domain_weights": domain_weights,
    }
    return sample_weights, diagnostics


def reweight_domain_losses(
    losses: torch.Tensor,
    domain_weights: torch.Tensor,
) -> torch.Tensor:
    if domain_weights.ndim == 1:
        if domain_weights.shape[0] != losses.shape[0]:
            raise ValueError("样本权重长度必须与 batch 大小一致")
        return losses * domain_weights.to(device=losses.device, dtype=losses.dtype).unsqueeze(-1)
    if domain_weights.shape != losses.shape:
        raise ValueError("逐 token 权重必须与 losses 等形")
    return losses * domain_weights.to(device=losses.device, dtype=losses.dtype)


@register_distillation_loss(DistillationLossSettings(names=["sampled_reverse_kl"], use_estimator=True))
def compute_sampled_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict[str, Any],
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del model_output
    if "distillation_advantages" not in data.keys():
        raise ValueError("缺少逐 token 的 distillation_advantages 字段")
    advantages = data["distillation_advantages"]
    if advantages.is_nested:
        advantages = advantages.to_padded_tensor(0.0)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    if advantages.shape != response_mask.shape:
        raise ValueError("蒸馏 advantage 必须与 response_mask 等形")

    advantages = advantages.to(device=response_mask.device, dtype=torch.float32)
    valid = response_mask.bool()
    metrics: dict[str, Any] = {
        "distillation/abs_advantage": advantages[valid].abs().mean().item() if valid.any() else 0.0,
        "distillation/clipped_fraction": ((advantages.abs() >= 3.0) & valid).sum().item()
        / max(int(valid.sum().item()), 1),
    }

    # 4.4 教师校准位移的 token 级可靠性权重（经 TQ 张量字段传入）
    effective_advantages = advantages
    if "distillation_token_weights" in data.keys():
        token_weights = data["distillation_token_weights"]
        if token_weights.is_nested:
            token_weights = token_weights.to_padded_tensor(1.0)
        if token_weights.shape != response_mask.shape:
            raise ValueError("distillation_token_weights 必须与 response_mask 等形")
        token_weights = token_weights.to(device=response_mask.device, dtype=torch.float32)
        effective_advantages = advantages * token_weights
        metrics["distillation/mean_token_reliability_weight"] = (
            token_weights[valid].mean().item() if valid.any() else 1.0
        )

    losses = -effective_advantages

    reweight_enabled = True
    if distillation_config is not None and hasattr(distillation_config, "get"):
        reweight_enabled = bool(distillation_config.get("domain_token_reweight", True))
    elif config is not None and hasattr(config, "get"):
        reweight_enabled = bool(config.get("domain_token_reweight", True))

    if reweight_enabled:
        if "distillation_domain_weights" in data.keys():
            weights = data["distillation_domain_weights"]
            if weights.is_nested:
                weights = weights.to_padded_tensor(1.0)
            losses = reweight_domain_losses(losses, weights)
        elif "domain_names" in data.keys() and "target_domain_shares" in data.keys():
            sample_weights, diagnostics = compute_domain_token_weights(
                effective_advantages,
                response_mask,
                list(data["domain_names"]),
                dict(data["target_domain_shares"]),
            )
            losses = reweight_domain_losses(losses, sample_weights)
            for domain, gap in diagnostics["share_gaps"].items():
                metrics[f"distillation/gradient_share_gap/{domain}"] = gap

    return losses, metrics
