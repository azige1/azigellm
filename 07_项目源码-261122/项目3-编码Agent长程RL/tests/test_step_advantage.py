"""测试 3.3 与 3.4 节：多段轨迹权重归一、GiGPO step 级分组与组合 advantage。"""

from __future__ import annotations

import math

import torch

from long_horizon_rl.step_advantage import (
    StepTokenSpan,
    build_combined_token_advantages,
    compute_discounted_step_returns,
    compute_gigpo_step_advantages,
)


def test_discounted_step_returns_gamma_095() -> None:
    # 终局 reward 0.70，倒数第 6 步折扣回报约为 0.70 * 0.95^6 ≈ 0.5146
    returns = compute_discounted_step_returns(0.70, num_steps=6, gamma=0.95)
    assert len(returns) == 6
    assert math.isclose(returns[0], 0.70 * (0.95**6), rel_tol=1e-6)
    assert math.isclose(returns[-1], 0.70 * 0.95, rel_tol=1e-6)


def test_gigpo_grouping_per_uid_and_singleton_zero() -> None:
    anchor = "KeyError: <str> in parse()"
    # 4 条轨迹都在该报错 anchor 上产生决策（同一 prompt uid）
    spans_0 = [StepTokenSpan(0, 0, 5, anchor, 0)] + [StepTokenSpan(0, 5, 6, None, i) for i in range(1, 6)]
    spans_1 = [StepTokenSpan(0, 0, 5, anchor, 0)] + [StepTokenSpan(0, 5, 6, None, i) for i in range(1, 14)]
    spans_2 = [StepTokenSpan(0, 0, 5, anchor, 0)] + [StepTokenSpan(0, 5, 6, None, i) for i in range(1, 5)]
    spans_3 = [StepTokenSpan(0, 0, 5, "UniqueError in other()", 0)]  # 单元素组

    adv_maps, stats = compute_gigpo_step_advantages(
        session_uids=["u1", "u1", "u1", "u1"],
        session_rewards=[0.70, 0.40, 0.74, 0.30],
        session_steps=[spans_0, spans_1, spans_2, spans_3],
        gamma=0.95,
        min_group_size=2,
    )
    # 步数最短且终局最高（5 步, 0.74）为正，步数最长（14 步, 0.40）为负
    assert adv_maps[2][(0, 0, 5)] > 0.0
    assert adv_maps[1][(0, 0, 5)] < 0.0
    # 单元素组 step advantage 记 0
    assert adv_maps[3][(0, 0, 5)] == 0.0
    assert stats["gigpo/num_step_groups"] == 2.0
    assert math.isclose(stats["gigpo/singleton_group_ratio"], 0.5, rel_tol=1e-6)


def test_multi_segment_trajectory_weights_and_combined_advantage() -> None:
    # 对照 3.3 节数值示例：轨迹 A 分 3 段（token 数 98、76、61 -> 比例与 9800/7600/6100 一致），
    # 轨迹 B 仅 1 段（token 数 100）
    lens = [98, 76, 61, 100]
    max_len = max(lens)
    response_mask = torch.zeros((4, max_len), dtype=torch.int64)
    for row, ln in enumerate(lens):
        response_mask[row, :ln] = 1

    base_adv = torch.zeros((4, max_len), dtype=torch.float32)
    base_adv[:3] = 1.0 * response_mask[:3]
    base_adv[3] = -1.0 * response_mask[3]

    batch_keys = ["u1_0_0", "u1_0_1", "u1_0_2", "u1_1_0"]
    session_spans = {
        "u1_0": [StepTokenSpan(segment_idx=0, start_token=10, end_token=30, anchor="KeyError in parse()", step_index=0)],
        "u1_1": [StepTokenSpan(segment_idx=0, start_token=10, end_token=30, anchor="KeyError in parse()", step_index=0)],
    }

    combined, weights, diag = build_combined_token_advantages(
        base_advantages=base_adv,
        response_mask=response_mask,
        batch_keys=batch_keys,
        session_rewards_by_key={"u1_0": 0.8, "u1_1": 0.2},
        session_spans_by_session=session_spans,
        omega=0.5,
        gamma=0.95,
    )

    # 验证 3.3 节归一特性：同一轨迹各段 token 权重完全相等，且轨迹 A 总权重 == 轨迹 B 总权重
    w_seg0 = weights[0, 0].item()
    w_seg1 = weights[1, 0].item()
    w_seg2 = weights[2, 0].item()
    assert math.isclose(w_seg0, w_seg1, rel_tol=1e-5)
    assert math.isclose(w_seg1, w_seg2, rel_tol=1e-5)
    total_w_traj_a = weights[:3].sum().item()
    total_w_traj_b = weights[3].sum().item()
    assert math.isclose(total_w_traj_a, total_w_traj_b, rel_tol=1e-4)

    # 验证 3.4 节：anchor 区间叠加了 +0.5 * A^step，非 anchor 区间保持轨迹级 advantage
    assert combined[0, 15].item() > 1.0
    assert math.isclose(combined[0, 0].item(), 1.0, rel_tol=1e-6)
    assert diag["curriculum/zero_variance_group_ratio"] == 0.0


def test_gigpo_groups_across_batch_unless_grouped_by_uid() -> None:
    anchor = "KeyError: <str> in parse()"
    spans_a = [StepTokenSpan(0, 0, 5, anchor, 0), StepTokenSpan(0, 5, 6, None, 1)]
    spans_b = [StepTokenSpan(0, 0, 5, anchor, 0)] + [StepTokenSpan(0, 5, 6, None, i) for i in range(1, 8)]

    # 默认按整个 batch 分组：不同题目的相同规范化报错归为一组
    adv_maps, stats = compute_gigpo_step_advantages(
        session_uids=["u1", "u2"],
        session_rewards=[0.8, 0.3],
        session_steps=[spans_a, spans_b],
    )
    assert adv_maps[0][(0, 0, 5)] > 0.0
    assert adv_maps[1][(0, 0, 5)] < 0.0
    assert stats["gigpo/num_step_groups"] == 1.0

    # group_by_uid=True 时两题各自成单元素组，step advantage 记 0
    adv_maps, stats = compute_gigpo_step_advantages(
        session_uids=["u1", "u2"],
        session_rewards=[0.8, 0.3],
        session_steps=[spans_a, spans_b],
        group_by_uid=True,
    )
    assert adv_maps[0][(0, 0, 5)] == 0.0
    assert stats["gigpo/num_step_groups"] == 2.0
