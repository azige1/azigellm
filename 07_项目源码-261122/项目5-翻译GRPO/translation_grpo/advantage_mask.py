"""5.3 Reference-anchor GRPO advantage shaping and 5.4 hard mask + gradient isolation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass
class AnchorAdvantageConfig:
    """Configuration for 5.3 anchor-corrected GRPO advantage and 5.4 hard masking."""

    anchor_enabled: bool = True
    anchor_lambda: float = 0.3
    hard_mask_enabled: bool = True
    min_valid_group_size: int = 2
    norm_adv_by_std_in_grpo: bool = True
    epsilon: float = 1e-6


def compute_translation_grpo_advantage(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Sequence[Any],
    *,
    invalid_mask: torch.Tensor | None = None,
    anchor_rewards: torch.Tensor | None = None,
    is_anchor_mask: torch.Tensor | None = None,
    is_padding_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    config: AnchorAdvantageConfig | None = None,
) -> dict[str, Any]:
    """Compute group-relative GRPO advantages with 5.3 anchor correction and 5.4 hard mask.

    Args:
        rewards: Scalar reward per trajectory, shape (bsz,).
        response_mask: Token-level response mask, shape (bsz, resp_len).
        uids: Group identifier per trajectory of length bsz.
        invalid_mask: Boolean mask (bsz,) marking repetition/overlength invalid outputs.
        anchor_rewards: Per-trajectory reference-anchor reward (bsz,), NaN when unavailable.
        is_anchor_mask: Boolean mask (bsz,) if any reference-anchor rows appear in the batch;
            such rows are excluded from group mean/std and masked out from loss.
        is_padding_mask: Boolean mask (bsz,) for upsampling padding rows.
        loss_mask: Optional token-level loss mask, shape (bsz, resp_len).
        config: AnchorAdvantageConfig.

    Returns:
        Dictionary containing advantages, returns, response_mask, loss_mask, scalar_advantages,
        and diagnostic metrics.
    """
    cfg = config or AnchorAdvantageConfig()
    bsz, resp_len = response_mask.shape
    device = response_mask.device
    dtype = rewards.dtype if rewards.is_floating_point() else torch.float32

    rewards_f = rewards.to(device=device, dtype=torch.float32)
    base_resp_mask = response_mask.clone()
    base_loss_mask = loss_mask.clone() if loss_mask is not None else response_mask.clone()

    invalid_b = (
        invalid_mask.to(device=device, dtype=torch.bool)
        if invalid_mask is not None
        else torch.zeros(bsz, device=device, dtype=torch.bool)
    )
    anchor_row_b = (
        is_anchor_mask.to(device=device, dtype=torch.bool)
        if is_anchor_mask is not None
        else torch.zeros(bsz, device=device, dtype=torch.bool)
    )
    padding_b = (
        is_padding_mask.to(device=device, dtype=torch.bool)
        if is_padding_mask is not None
        else torch.zeros(bsz, device=device, dtype=torch.bool)
    )

    if not cfg.hard_mask_enabled:
        effective_invalid_b = torch.zeros(bsz, device=device, dtype=torch.bool)
    else:
        effective_invalid_b = invalid_b

    # Policy candidates eligible for group statistics and policy loss.
    eligible_b = (~padding_b) & (~anchor_row_b) & (~effective_invalid_b)

    groups: dict[Any, list[int]] = defaultdict(list)
    all_policy_groups: dict[Any, list[int]] = defaultdict(list)
    for i in range(bsz):
        if padding_b[i] or anchor_row_b[i]:
            continue
        uid = uids[i]
        all_policy_groups[uid].append(i)
        if eligible_b[i]:
            groups[uid].append(i)

    # Resolve per-sample anchor reward if anchor rows are inside the batch or provided separately.
    resolved_anchor = torch.full((bsz,), float("nan"), device=device, dtype=torch.float32)
    if anchor_rewards is not None:
        resolved_anchor = anchor_rewards.to(device=device, dtype=torch.float32).clone()

    if anchor_row_b.any():
        uid_to_anchor_val: dict[Any, float] = {}
        for i in range(bsz):
            if anchor_row_b[i] and not padding_b[i]:
                uid_to_anchor_val[uids[i]] = float(rewards_f[i].item())
        for i in range(bsz):
            if torch.isnan(resolved_anchor[i]) and uids[i] in uid_to_anchor_val:
                resolved_anchor[i] = uid_to_anchor_val[uids[i]]

    scalar_adv = torch.zeros(bsz, device=device, dtype=torch.float32)
    skipped_group_mask = torch.zeros(bsz, device=device, dtype=torch.bool)
    skipped_groups = 0

    for uid, policy_indices in all_policy_groups.items():
        valid_indices = groups.get(uid, [])
        if len(valid_indices) < cfg.min_valid_group_size:
            skipped_groups += 1
            for idx in policy_indices:
                skipped_group_mask[idx] = True
            continue

        idx_t = torch.tensor(valid_indices, device=device, dtype=torch.long)
        grp_rewards = rewards_f[idx_t]
        mu_g = grp_rewards.mean()
        sigma_g = grp_rewards.std(unbiased=True) if len(valid_indices) > 1 else torch.tensor(0.0, device=device)

        if cfg.norm_adv_by_std_in_grpo:
            normed = (grp_rewards - mu_g) / (sigma_g + cfg.epsilon)
        else:
            normed = grp_rewards - mu_g
        scalar_adv[idx_t] = normed

    active_b = eligible_b & (~skipped_group_mask)

    # 5.3 Reference-anchor correction: + lambda * tanh((r_i - r_anc) / sigma_B)
    anchor_delta = torch.zeros(bsz, device=device, dtype=torch.float32)
    below_anchor_groups = 0
    total_active_groups = 0
    sigma_b_val = 0.0

    if cfg.anchor_enabled and cfg.anchor_lambda != 0.0:
        has_anc_b = active_b & (~torch.isnan(resolved_anchor))
        if has_anc_b.any():
            diff = rewards_f[has_anc_b] - resolved_anchor[has_anc_b]
            sigma_b = diff.std(unbiased=False) if diff.numel() > 1 else torch.tensor(1.0, device=device)
            sigma_b = torch.clamp(sigma_b, min=cfg.epsilon)
            sigma_b_val = float(sigma_b.item())
            anchor_delta[has_anc_b] = cfg.anchor_lambda * torch.tanh(diff / sigma_b)
            scalar_adv[has_anc_b] = scalar_adv[has_anc_b] + anchor_delta[has_anc_b]

        for uid, valid_indices in groups.items():
            if len(valid_indices) < cfg.min_valid_group_size:
                continue
            idx_t = torch.tensor(valid_indices, device=device, dtype=torch.long)
            anc_vals = resolved_anchor[idx_t]
            valid_anc = anc_vals[~torch.isnan(anc_vals)]
            if valid_anc.numel() > 0:
                total_active_groups += 1
                if float(rewards_f[idx_t].max().item()) < float(valid_anc[0].item()):
                    below_anchor_groups += 1

    # 5.4 Hard mask: isolate invalid outputs, anchor rows, padding rows, and skipped groups.
    keep_b = active_b.to(dtype=base_resp_mask.dtype).unsqueeze(-1)
    masked_response_mask = base_resp_mask * keep_b
    masked_loss_mask = base_loss_mask * keep_b

    token_advantages = scalar_adv.unsqueeze(-1) * masked_response_mask.to(dtype=torch.float32)
    token_returns = token_advantages.clone()

    # Diagnostic: compare advantage mean of valid samples in groups with vs without invalid samples.
    adv_in_groups_with_invalid: list[float] = []
    adv_in_clean_groups: list[float] = []
    for uid, policy_indices in all_policy_groups.items():
        if len(groups.get(uid, [])) < cfg.min_valid_group_size:
            continue
        has_inv = any(bool(invalid_b[idx].item()) for idx in policy_indices)
        valid_vals = [float(scalar_adv[idx].item()) for idx in groups[uid]]
        if has_inv:
            adv_in_groups_with_invalid.extend(valid_vals)
        else:
            adv_in_clean_groups.extend(valid_vals)

    non_padding_policy = (~padding_b) & (~anchor_row_b)
    n_policy = int(non_padding_policy.sum().item())
    n_invalid = int((invalid_b & non_padding_policy).sum().item())

    metrics: dict[str, float] = {
        "grpo/invalid_sample_count": float(n_invalid),
        "grpo/invalid_sample_ratio": float(n_invalid / max(1, n_policy)),
        "grpo/skipped_group_count": float(skipped_groups),
        "grpo/active_sample_count": float(int(active_b.sum().item())),
        "grpo/anchor_sigma_batch": float(sigma_b_val),
        "grpo/anchor_delta_abs_mean": float(anchor_delta[active_b].abs().mean().item()) if active_b.any() else 0.0,
        "grpo/below_anchor_group_ratio": float(below_anchor_groups / max(1, total_active_groups)),
        "grpo/adv_mean_groups_with_invalid": (
            float(sum(adv_in_groups_with_invalid) / len(adv_in_groups_with_invalid))
            if adv_in_groups_with_invalid
            else 0.0
        ),
        "grpo/adv_mean_clean_groups": (
            float(sum(adv_in_clean_groups) / len(adv_in_clean_groups)) if adv_in_clean_groups else 0.0
        ),
    }

    return {
        "advantages": token_advantages.to(dtype=dtype),
        "returns": token_returns.to(dtype=dtype),
        "response_mask": masked_response_mask,
        "loss_mask": masked_loss_mask,
        "scalar_advantages": scalar_adv,
        "active_mask": active_b,
        "metrics": metrics,
    }


def verify_zero_grad_on_masked_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    param_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> float:
    """Run one backward pass on an all-masked micro-batch and return max/L2 LoRA grad norm.

    Implements the 5.4 periodic gradient-isolation check: invalid samples must leave
    LoRA parameter gradients strictly at zero.
    """
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    resp_len = response_mask.size(1)
    # Next-token log probabilities aligned with response tokens.
    shift_logits = logits[:, -resp_len - 1 : -1, :]
    shift_targets = input_ids[:, -resp_len:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1).gather(
        dim=-1, index=shift_targets.unsqueeze(-1)
    ).squeeze(-1)

    ratio = torch.exp(torch.clamp(log_probs - old_log_probs.float(), min=-20.0, max=20.0))
    pg_losses = -advantages.float() * ratio
    mask_f = response_mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    loss = (pg_losses * mask_f).sum() / denom
    loss.backward()

    sq_norm = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if param_filter is not None and not param_filter(name, param):
            continue
        sq_norm += float(param.grad.detach().float().pow(2).sum().item())

    model.zero_grad(set_to_none=True)
    return float(sq_norm**0.5)
