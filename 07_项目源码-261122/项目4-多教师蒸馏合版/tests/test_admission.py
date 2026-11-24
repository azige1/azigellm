import json

import pytest

from multi_teacher_distillation.admission import (
    evaluate_admission,
    paired_bootstrap_score_diff_lower_bound,
    same_model_source,
)


def _write_fake_checkpoint(path, token="token"):
    path.mkdir()
    (path / "tokenizer.json").write_text(json.dumps({"vocab": {token: 0}}), encoding="utf-8")
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 151936}),
        encoding="utf-8",
    )


def test_admission_checks_gain_general_regression_tokenizer_and_lb(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _write_fake_checkpoint(student)
    _write_fake_checkpoint(teacher)

    result = evaluate_admission("code_gen", 58.8, 64.5, 70.0, 69.5, student, teacher, score_diff_lower_bound=3.2)
    assert result.admitted
    assert result.teacher_reliability_multiplier == pytest.approx(1.0)

    low_lb = evaluate_admission(
        "instruction_following", 83.5, 87.8, 70.0, 69.6, student, teacher, score_diff_lower_bound=1.6
    )
    assert low_lb.admitted
    assert low_lb.teacher_reliability_multiplier == pytest.approx(0.8)

    too_low = evaluate_admission("code_gen", 58.8, 60.0, 70.0, 69.5, student, teacher)
    assert not too_low.admitted

    (teacher / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert not same_model_source(student, teacher)


def test_paired_bootstrap_lower_bound_is_positive_for_consistent_gain():
    start_items = [0.0] * 50 + [1.0] * 50
    teacher_items = [0.0] * 35 + [1.0] * 65
    lb = paired_bootstrap_score_diff_lower_bound(start_items, teacher_items, bootstrap_samples=500, seed=17)
    assert lb > 5.0
