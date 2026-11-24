import pytest

from multi_teacher_distillation.rollout_masking import (
    advantage_deciles,
    clipped_advantage,
    is_truncated,
    repeated_ngram_fraction,
    rollout_loss_mask,
    summarize_rollouts,
)


def test_truncation_requires_limit_and_missing_terminal_token():
    assert is_truncated([1, 2], 2, {9})
    assert not is_truncated([1, 9], 2, {9})
    assert not is_truncated([1], 2, {9})
    assert rollout_loss_mask([1, 2], 2, {9}) == [0, 0]


def test_advantage_is_clipped_to_plus_minus_five_by_default():
    values = clipped_advantage([0.2, -8.0, 8.0], [0.1, 0.0, 0.0])
    assert values.tolist() == pytest.approx([0.1, -5.0, 5.0])


def test_repetition_and_domain_warning():
    rows = [
        {"domain": "issue_fix", "response_ids": [1, 1, 1, 1, 1], "truncated": True},
        {"domain": "issue_fix", "response_ids": [1, 2, 3, 4], "truncated": False},
    ]
    result = summarize_rollouts(rows)
    assert result["domains"]["issue_fix"]["truncation_rate"] == 0.5
    assert result["warning_domains"] == ["issue_fix"]
    assert repeated_ngram_fraction([1, 1, 1, 1], 4) == 0.5


def test_advantage_deciles_have_ten_groups():
    rows = [
        {"mean_advantage": index, "correct": index % 2, "length": 10 + index,
         "at_limit": False, "repetition_rate": 0.0}
        for index in range(20)
    ]
    groups = advantage_deciles(rows)
    assert len(groups) == 10
    assert all(group["count"] == 2 for group in groups)
