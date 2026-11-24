import pytest

from multi_teacher_distillation.termination import require_termination_alignment


def test_termination_gap_above_two_is_fatal():
    report = {"templates": {"passed": True}, "logprob_checks": [{"teacher": "report", "gap": 2.1}]}
    with pytest.raises(RuntimeError):
        require_termination_alignment(report)
