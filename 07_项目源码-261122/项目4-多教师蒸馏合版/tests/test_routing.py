import pytest

from multi_teacher_distillation.routing import (
    check_routing_share_drift,
    filter_cached_logprobs_by_version,
    propagate_segment_metadata,
    validate_routing_stage,
)


def test_segment_metadata_propagation_and_four_stage_validation():
    parent = {"domain": "repo_impl", "teacher": "repo_impl", "teacher_version": "ckpt_v1"}
    seg0 = propagate_segment_metadata({}, parent, segment_index=0)
    seg1 = propagate_segment_metadata({}, parent, segment_index=1)
    assert seg1["domain"] == "repo_impl"
    assert seg1["teacher"] == "repo_impl"
    assert seg1["teacher_version"] == "ckpt_v1"
    assert seg1["segment_index"] == 1

    for stage in ("data_build", "rollout_output", "teacher_request", "pre_loss"):
        audit = validate_routing_stage(stage, [seg0, seg1], strict_mode="raise")
        assert audit["valid_count"] == 2
        assert audit["dropped_count"] == 0

    missing_domain = {"teacher": "self_anchor", "segment_index": 0}
    with pytest.raises(RuntimeError):
        validate_routing_stage("rollout_output", [missing_domain], strict_mode="raise")

    drop_audit = validate_routing_stage("rollout_output", [seg0, missing_domain], strict_mode="drop")
    assert drop_audit["valid_indices"] == [0]
    assert drop_audit["dropped_indices"] == [1]
    assert drop_audit["dropped_count"] == 1


def test_version_mismatch_invalidates_cached_logprobs_and_share_drift_alarm():
    rows = [
        {
            "domain": "issue_fix",
            "teacher": "issue_fix",
            "teacher_version": "old_v1",
            "segment_index": 0,
            "cached_teacher_version": "old_v1",
            "cached_teacher_logprobs": [-0.2, -0.3],
        },
        {
            "domain": "code_gen",
            "teacher": "code_gen",
            "teacher_version": "cg_v1",
            "segment_index": 0,
            "cached_teacher_version": "cg_v1",
            "cached_teacher_logprobs": [-0.1, -0.1],
        },
    ]
    res = filter_cached_logprobs_by_version(rows, {"issue_fix": "new_v2", "code_gen": "cg_v1"})
    assert res["cache_hits"] == 1
    assert res["recomputed"] == 1
    assert "cached_teacher_logprobs" not in rows[0]
    assert rows[0]["needs_logprob_recompute"] is True
    assert rows[1]["cached_teacher_logprobs"] == [-0.1, -0.1]

    drift = check_routing_share_drift(
        {"repo_impl": 160, "self_anchor": 100},
        {"repo_impl": 0.18, "self_anchor": 0.08},
        drift_threshold=0.01,
    )
    assert drift["alarm"] is True
    assert "repo_impl" in drift["alarm_domains"]
