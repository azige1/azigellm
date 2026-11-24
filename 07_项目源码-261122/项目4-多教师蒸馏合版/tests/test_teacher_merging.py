import json
from pathlib import Path

from multi_teacher_distillation.teacher_merging import build_teacher_pool_from_matrix


def test_ability_matrix_merges_ten_candidates_into_eight_teachers():
    matrix_path = Path(__file__).resolve().parents[1] / "data" / "examples" / "ability_matrix.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))

    report = build_teacher_pool_from_matrix(
        score_matrix=payload["score_matrix"],
        domain_stds=payload["domain_stds"],
        reward_sources=payload["reward_sources"],
        start_subclass_scores=payload["start_subclass_scores"],
        teacher_subclass_scores=payload["teacher_subclass_scores"],
        noise_multiplier=2.0,
    )
    assert report["merged_teacher_total"] == 8
    mapping = report["domain_to_teacher"]
    assert mapping["func_gen"] == "code_gen"
    assert mapping["competitive_coding"] == "code_gen"
    assert mapping["math"] == "math_science"
    assert mapping["science"] == "math_science"
    assert mapping["repo_impl"] == "repo_impl"
    assert mapping["issue_fix"] == "issue_fix"
    assert mapping["self_anchor"] == "self_anchor"
    merged_flags = {item["teacher_id"]: item["retrain_required"] for item in report["teachers"]}
    assert merged_flags["code_gen"] is True
    assert merged_flags["math_science"] is True
    assert merged_flags["repo_impl"] is False


def test_subclass_below_start_triggers_split():
    stds = {"a": 0.5, "b": 0.5}
    matrix = {"a": {"a": 70.0, "b": 50.0}, "b": {"a": 50.0, "b": 70.0}}
    sources = {"a": "execution_test", "b": "rule_verifier"}
    start_subs = {"b": {"sub1": 60.0, "sub2": 65.0}}
    teacher_subs = {"b": {"sub1": 72.0, "sub2": 62.0}}  # sub2 < start

    report = build_teacher_pool_from_matrix(
        score_matrix=matrix,
        domain_stds=stds,
        reward_sources=sources,
        start_subclass_scores=start_subs,
        teacher_subclass_scores=teacher_subs,
    )
    assert report["merged_teacher_total"] == 3
    assert report["domain_to_teacher"]["b:sub1"] == "b__sub1"
    assert report["domain_to_teacher"]["b:sub2"] == "b__sub2"
