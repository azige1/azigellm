import numpy as np
import pytest
import torch

from multi_teacher_distillation.reliability_weighting import (
    CalibrationManager,
    build_batch_reliability_weights,
    fit_calibration_table,
    score_token_reliability_weights,
)


def test_calibration_bucketing_downweights_failure_correlated_high_shift_tokens():
    # 100 tokens: shifts 0..99; highest 10% (shifts >= 90) mostly from failed trajectories
    shifts = list(range(100))
    passed = [1.0 if (idx < 90 and idx % 2 == 0) or (idx >= 90 and idx == 99) else 0.0 for idx in range(100)]

    table = fit_calibration_table(
        "issue_fix",
        shifts=shifts,
        passed_flags=passed,
        num_bins=10,
        alpha=2.0,
        lift_cap=1.5,
        admission_lb=2.5,  # >= 2 -> multiplier 1.0
    )
    assert table.admission_multiplier == pytest.approx(1.0)
    assert table.bucket_weights[-1] < table.bucket_weights[4]

    weights = score_token_reliability_weights(
        np.asarray([50.0, 95.0]),
        table,
        response_mask=np.asarray([1, 1]),
        normalize_within_domain=True,
    )
    assert np.mean(weights) == pytest.approx(1.0)
    assert weights[1] < weights[0]


def test_batch_reliability_weights_apply_admission_multiplier_and_refresh_schedule():
    shifts = list(range(100))
    passed = [1.0 if idx % 2 == 0 else 0.0 for idx in range(100)]
    if_table = fit_calibration_table(
        "instruction_following",
        shifts=shifts,
        passed_flags=passed,
        admission_lb=1.6,  # 1.6 / 2.0 = 0.8
    )
    assert if_table.admission_multiplier == pytest.approx(0.8)

    t_lps = torch.tensor([[float(x) for x in [10, 50, 90]], [0.0, 0.0, 0.0]])
    r_lps = torch.zeros_like(t_lps)
    masks = torch.tensor([[1, 1, 1], [1, 1, 0]])
    weights, means = build_batch_reliability_weights(
        ["instruction_following", "self_anchor"],
        t_lps,
        r_lps,
        masks,
        {"instruction_following": if_table},
        normalize_within_domain=True,
    )
    assert means["instruction_following"] == pytest.approx(0.8)
    assert weights[0].mean().item() == pytest.approx(0.8, abs=1e-4)
    assert weights[1, :2].tolist() == pytest.approx([1.0, 1.0])
    assert weights[1, 2].item() == pytest.approx(0.0)

    manager = CalibrationManager(tables={"instruction_following": if_table}, refresh_interval=40)
    assert manager.should_refresh(40) is True
    assert manager.should_refresh(39) is False
