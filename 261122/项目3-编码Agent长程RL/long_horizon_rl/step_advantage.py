r"""GiGPO step 级 advantage 与多段轨迹 token 归一权重计算。

包含两项核心计算：
1. 3.3 节多段轨迹按总 token 数归一的权重：同一轨迹各段 token 共享相同的总 token 计数
   $T_\tau = \sum_k |y_k|$，权重为 $\frac{\bar{T}}{T_\tau}$，避免 4 段轨迹权重为 1 段轨迹的 4 倍。
2. 3.4 节 GiGPO step 级分组 advantage：同一 prompt uid 内，规范化报错 anchor 相同的步
   组成 step 组，以终局 reward 折扣回报 $R_t = \gamma^{N-t} r_{\text{final}}$ 在组内标准化得到
   $A^{\text{step}}_t$，最终 token advantage 为 $A_t = A_\tau + \omega A^{\text{step}}_t$。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class StepTokenSpan:
    r"""单个工具交互步在某一段响应序列中的 token 区间与规范化报错 anchor。"""

    segment_idx: int
    start_token: int
    end_token: int
    anchor: str | None
    step_index: int


def compute_discounted_step_returns(
    terminal_reward: float,
    num_steps: int,
    gamma: float = 0.95,
) -> list[float]:
    r"""计算每一步到终局的折扣回报 $R_t = \gamma^{N - t} r_{\text{final}}$。

    约定最后一步（idx = num_steps - 1）之后即终局，距终局步数为 1 时回报为 $\gamma^1 r$，
    或若步编号从 0..num_steps-1 计且剩余步数 `steps_remaining = num_steps - step_idx`，
    则 $R_t = \gamma^{\text{steps\_remaining}} r_{\text{final}}$（与 3.4 数值演算示例一致）。
    """
    if num_steps <= 0:
        return []
    return [float(terminal_reward) * (gamma ** (num_steps - idx)) for idx in range(num_steps)]


def compute_gigpo_step_advantages(
    *,
    session_uids: Sequence[str],
    session_rewards: Sequence[float],
    session_steps: Sequence[Sequence[StepTokenSpan]],
    gamma: float = 0.95,
    min_group_size: int = 2,
    eps: float = 1e-6,
    group_by_uid: bool = False,
) -> tuple[list[dict[tuple[int, int, int], float]], dict[str, float]]:
    r"""按 prompt uid 与规范化报错 anchor 分组，计算每个 step span 的 step 级 advantage。

    返回：
    - 每条轨迹一个字典，映射 ``(segment_idx, start_token, end_token) -> step_advantage``；
    - 诊断指标字典：``gigpo/singleton_group_ratio``、``gigpo/num_step_groups``、
      ``gigpo/valid_step_fraction``。
    """
    # 默认在整个 batch 内按规范化报错分组；group_by_uid=True 时只在同题组内比较
    uid_buckets: dict[str, list[int]] = defaultdict(list)
    for traj_idx, uid in enumerate(session_uids):
        uid_buckets[str(uid) if group_by_uid else "batch"].append(traj_idx)

    per_traj_adv: list[dict[tuple[int, int, int], float]] = [{} for _ in session_uids]
    total_groups = 0
    singleton_groups = 0
    total_anchored_steps = 0
    non_singleton_steps = 0

    for uid, traj_indices in uid_buckets.items():
        # anchor -> list of (traj_idx, span, discounted_return)
        anchor_groups: dict[str, list[tuple[int, StepTokenSpan, float]]] = defaultdict(list)
        for traj_idx in traj_indices:
            spans = session_steps[traj_idx]
            returns = compute_discounted_step_returns(
                session_rewards[traj_idx],
                num_steps=len(spans),
                gamma=gamma,
            )
            for span, ret in zip(spans, returns, strict=True):
                if span.anchor:
                    total_anchored_steps += 1
                    anchor_groups[span.anchor].append((traj_idx, span, ret))

        for anchor, entries in anchor_groups.items():
            total_groups += 1
            if len(entries) < min_group_size:
                singleton_groups += 1
                for traj_idx, span, _ in entries:
                    per_traj_adv[traj_idx][(span.segment_idx, span.start_token, span.end_token)] = 0.0
                continue

            non_singleton_steps += len(entries)
            vals = np.array([e[2] for e in entries], dtype=np.float64)
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            for (traj_idx, span, ret) in entries:
                if std < eps:
                    adv = 0.0
                else:
                    adv = (ret - mean) / (std + eps)
                per_traj_adv[traj_idx][(span.segment_idx, span.start_token, span.end_token)] = float(adv)

    singleton_ratio = float(singleton_groups / total_groups) if total_groups > 0 else 0.0
    valid_step_fraction = float(non_singleton_steps / total_anchored_steps) if total_anchored_steps > 0 else 0.0
    stats = {
        "gigpo/singleton_group_ratio": singleton_ratio,
        "gigpo/num_step_groups": float(total_groups),
        "gigpo/valid_step_fraction": valid_step_fraction,
    }
    return per_traj_adv, stats


def build_combined_token_advantages(
    *,
    base_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    batch_keys: Sequence[str],
    session_rewards_by_key: dict[str, float],
    session_spans_by_session: dict[str, list[StepTokenSpan]],
    omega: float = 0.5,
    gamma: float = 0.95,
    min_group_size: int = 2,
    group_by_uid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    r"""将轨迹级 advantage 与 step 级 advantage 叠加，并计算按轨迹总 token 归一的权重。

    Args:
        base_advantages: ``(bsz, response_len)``，由 ``compute_advantage_for_multi_trajectories``
            得到的轨迹级 GRPO advantage。
        response_mask: ``(bsz, response_len)``，1 为模型生成 token（含摘要），0 为环境/padding。
        batch_keys: TQ 键列表，格式 ``{uid}_{session_id}_{segment_idx}``。
        session_rewards_by_key: 映射 ``{uid}_{session_id} -> reward``。
        session_spans_by_session: 映射 ``{uid}_{session_id} -> list[StepTokenSpan]``。
        omega: step 级 advantage 权重 $\omega$（默认 0.5）。
        gamma: step 级折扣因子 $\gamma$（默认 0.95）。
        min_group_size: step 组最小样本数（默认 2，单元素组计 0）。

    Returns:
        ``(combined_advantages, traj_token_weights, metrics)``，形状均与 ``base_advantages`` 相同。
    """
    session_order: list[str] = []
    session_seen: set[str] = set()
    row_session_and_seg: list[tuple[str, int]] = []

    for key in batch_keys:
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            session_key = f"{parts[0]}_{parts[1]}"
            seg_idx = int(parts[2])
        else:
            session_key = key
            seg_idx = 0
        row_session_and_seg.append((session_key, seg_idx))
        if session_key not in session_seen:
            session_seen.add(session_key)
            session_order.append(session_key)

    session_uids = [s.rsplit("_", 1)[0] if "_" in s else s for s in session_order]
    session_rewards = [float(session_rewards_by_key.get(s, 0.0)) for s in session_order]
    session_steps = [session_spans_by_session.get(s, []) for s in session_order]

    per_traj_adv, gigpo_stats = compute_gigpo_step_advantages(
        session_uids=session_uids,
        session_rewards=session_rewards,
        session_steps=session_steps,
        gamma=gamma,
        min_group_size=min_group_size,
        group_by_uid=group_by_uid,
    )
    session_to_adv_map = {s: adv_map for s, adv_map in zip(session_order, per_traj_adv, strict=True)}

    device = base_advantages.device
    dtype = base_advantages.dtype
    step_adv_tensor = torch.zeros_like(base_advantages)
    seq_len = base_advantages.size(1)

    for row_idx, (session_key, seg_idx) in enumerate(row_session_and_seg):
        adv_map = session_to_adv_map.get(session_key, {})
        for (span_seg, start_tok, end_tok), step_adv in adv_map.items():
            if span_seg != seg_idx or step_adv == 0.0:
                continue
            lo = max(0, start_tok)
            hi = min(seq_len, end_tok)
            if lo < hi:
                step_adv_tensor[row_idx, lo:hi] = float(step_adv)

    mask_f = response_mask.to(dtype=dtype)
    combined = (base_advantages + float(omega) * step_adv_tensor) * mask_f

    # 3.3 节：按整条轨迹的总有效 assistant token 数归一
    session_token_totals: dict[str, float] = defaultdict(float)
    row_token_counts = mask_f.sum(dim=-1)
    for row_idx, (session_key, _) in enumerate(row_session_and_seg):
        session_token_totals[session_key] += float(row_token_counts[row_idx].item())

    num_sessions = max(1, len(session_order))
    total_batch_tokens = max(1.0, float(row_token_counts.sum().item()))
    traj_token_weights = torch.zeros_like(base_advantages)
    for row_idx, (session_key, _) in enumerate(row_session_and_seg):
        traj_tokens = session_token_totals.get(session_key, 0.0)
        if traj_tokens > 0.0:
            # 乘以 total_batch_tokens / num_sessions，使 token-mean 聚合等价于按轨迹等权平均
            weight = total_batch_tokens / (num_sessions * traj_tokens)
            traj_token_weights[row_idx] = float(weight)
    traj_token_weights = traj_token_weights * mask_f

    # 统计零方差组占比（以 prompt uid 为组）
    uid_rewards: dict[str, list[float]] = defaultdict(list)
    for uid, r in zip(session_uids, session_rewards, strict=True):
        uid_rewards[uid].append(r)
    zero_var_groups = sum(
        1 for vals in uid_rewards.values() if len(vals) > 1 and float(np.std(vals)) < 1e-8
    )
    num_uid_groups = sum(1 for vals in uid_rewards.values() if len(vals) > 1)
    gigpo_stats["curriculum/zero_variance_group_ratio"] = (
        float(zero_var_groups / num_uid_groups) if num_uid_groups > 0 else 0.0
    )

    return combined, traj_token_weights, gigpo_stats
