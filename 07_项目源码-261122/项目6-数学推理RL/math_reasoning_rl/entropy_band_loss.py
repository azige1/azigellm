"""Entropy bandwidth hinge regularization and patched PPO loss for verl 0.9.1 (6.2)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from math_reasoning_rl.config import EntropyBandConfig

# Hold reference to the original verl ppo_loss so wrapping is idempotent.
_ORIG_PPO_LOSS = None


def compute_entropy_band_hinge_loss(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    low: float = 0.45,
    high: float = 0.65,
    coef: float = 0.005,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one-sided hinge penalty when token-mean entropy leaves ``[low, high]``.

    Formula:
        L_ent = c * [max(0, H - H_hi) + max(0, H_lo - H)]

    When ``batch_num_tokens`` is provided during gradient accumulation over micro-batches,
    scales the micro-batch penalty by ``local_tokens / batch_num_tokens * dp_size`` so
    gradients sum across micro-batches consistently with global batch normalization.
    """
    mask_float = response_mask.to(dtype=entropy.dtype)
    local_tokens = mask_float.sum()
    if float(local_tokens.item()) <= 0.0:
        zero = entropy.sum() * 0.0
        return zero, {
            "actor/entropy_band_mean": 0.0,
            "actor/entropy_band_hinge_loss": 0.0,
            "actor/entropy_band_upper_violation": 0.0,
            "actor/entropy_band_lower_violation": 0.0,
            "actor/entropy_band_in_range": 1.0,
        }

    mean_entropy = (entropy * mask_float).sum() / local_tokens.clamp(min=1.0)
    upper_violation = F.relu(mean_entropy - high)
    lower_violation = F.relu(low - mean_entropy)
    raw_hinge = coef * (upper_violation + lower_violation)

    if batch_num_tokens is not None and batch_num_tokens > 0:
        token_share = (local_tokens / float(batch_num_tokens)) * float(dp_size)
        hinge_loss = raw_hinge * token_share
    else:
        hinge_loss = raw_hinge

    in_range = float((mean_entropy.detach() >= low) and (mean_entropy.detach() <= high))
    metrics = {
        "actor/entropy_band_mean": float(mean_entropy.detach().item()),
        "actor/entropy_band_hinge_loss": float(hinge_loss.detach().item()),
        "actor/entropy_band_upper_violation": float(upper_violation.detach().item()),
        "actor/entropy_band_lower_violation": float(lower_violation.detach().item()),
        "actor/entropy_band_in_range": in_range,
    }
    return hinge_loss, metrics


def set_original_ppo_loss(fn: Any) -> None:
    """Record the original ``verl.workers.utils.losses.ppo_loss`` function."""
    global _ORIG_PPO_LOSS
    if _ORIG_PPO_LOSS is None and fn is not entropy_band_ppo_loss:
        _ORIG_PPO_LOSS = fn


def entropy_band_ppo_loss(config: Any, model_output: dict[str, Any], data: TensorDict, dp_group=None):
    """Drop-in replacement for ``verl.workers.utils.losses.ppo_loss`` adding entropy-band hinge loss."""
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils import losses as verl_losses
    from verl.workers.utils.padding import no_padding_2_padding

    base_ppo_loss = _ORIG_PPO_LOSS or getattr(verl_losses, "_orig_ppo_loss", None)
    if base_ppo_loss is None:
        raise RuntimeError("Original verl ppo_loss reference has not been initialized before patching.")

    # Compute raw entropy before base_ppo_loss selects fields on `data`.
    raw_entropy = model_output.get("entropy", None)
    padded_entropy = no_padding_2_padding(raw_entropy, data) if raw_entropy is not None else None
    padded_mask = data.select("response_mask").to_padded_tensor()["response_mask"].to(bool)
    dp_size = int(data["dp_size"]) if "dp_size" in data else 1
    batch_num_tokens = data["batch_num_tokens"] if "batch_num_tokens" in data else None

    policy_loss, metrics = base_ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)

    band_cfg = EntropyBandConfig.from_env()
    if band_cfg.enabled and padded_entropy is not None:
        hinge_loss, band_metrics = compute_entropy_band_hinge_loss(
            entropy=padded_entropy,
            response_mask=padded_mask,
            low=band_cfg.low,
            high=band_cfg.high,
            coef=band_cfg.coef,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
        )
        policy_loss = policy_loss + hinge_loss

        use_sum_agg = (
            dp_size > 1
            or batch_num_tokens is not None
            or (data["global_batch_size"] if "global_batch_size" in data else None) is not None
            or getattr(config, "loss_scale_factor", None) is not None
        )
        hinge_agg = AggregationType.SUM if use_sum_agg else AggregationType.MEAN
        metrics["actor/entropy_band_hinge_loss"] = Metric(value=hinge_loss, aggregation=hinge_agg)
        for key in (
            "actor/entropy_band_mean",
            "actor/entropy_band_upper_violation",
            "actor/entropy_band_lower_violation",
            "actor/entropy_band_in_range",
        ):
            metrics[key] = Metric(value=band_metrics[key], aggregation=AggregationType.MEAN)

    return policy_loss, metrics
