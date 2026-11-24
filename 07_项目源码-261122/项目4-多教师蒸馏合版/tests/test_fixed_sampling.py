import pytest

from multi_teacher_distillation import DEFAULT_DOMAIN_SHARES, DEFAULT_POOL_SIZES, SELF_ANCHOR_DOMAIN
from multi_teacher_distillation.fixed_sampling import CrossStepQuota, FixedDomainSampler, distillation_shares


def test_default_pool_and_shares_match_eight_teachers_plus_self_anchor():
    assert sum(DEFAULT_POOL_SIZES.values()) == 96000
    assert len(DEFAULT_DOMAIN_SHARES) == 9
    assert DEFAULT_DOMAIN_SHARES[SELF_ANCHOR_DOMAIN] == pytest.approx(0.08)
    assert sum(DEFAULT_DOMAIN_SHARES.values()) == pytest.approx(1.0)
    assert distillation_shares() == DEFAULT_DOMAIN_SHARES


def test_cross_step_quota_preserves_long_run_counts():
    quota = CrossStepQuota({"a": 0.5, "b": 0.3, "c": 0.2})
    totals = {name: 0 for name in quota.shares}
    for _ in range(100):
        counts = quota.counts(7)
        assert sum(counts.values()) == 7
        for name, value in counts.items():
            totals[name] += value
    assert totals == {"a": 350, "b": 210, "c": 140}


def test_sampler_resume_restores_quota_and_rng():
    indices = {"a": [0, 1], "b": [2, 3]}
    sampler = FixedDomainSampler(indices, {"a": 0.5, "b": 0.5}, seed=3)
    sampler.sample_indices(5)
    state = sampler.state_dict()
    expected = sampler.sample_indices(7)
    restored = FixedDomainSampler(indices, {"a": 0.5, "b": 0.5}, seed=4)
    restored.load_state_dict(state)
    assert restored.sample_indices(7) == expected
