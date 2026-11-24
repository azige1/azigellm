import asyncio

from multi_teacher_distillation.domains.verifiers import (
    verify_execution_pass_rate,
    verify_instruction_following,
    verify_math_science,
    verify_tool_calling,
)


def test_math_science_verifier_handles_boxed_and_numeric():
    res = asyncio.run(verify_math_science({"answer": "5"}, "推导得 \\boxed{5}"))
    assert res.passed
    assert res.score == 1.0


def test_instruction_following_checks_strict_constraints():
    task = {
        "constraints": [
            {"type": "max_words", "limit": 10},
            {"type": "include_keyword", "keyword": "assertion"},
        ]
    }
    ok = asyncio.run(verify_instruction_following(task, "Every unit test needs an assertion."))
    bad = asyncio.run(verify_instruction_following(task, "Every unit test needs a check."))
    assert ok.passed and ok.score == 1.0
    assert not bad.passed and bad.score == 0.0


def test_tool_calling_verifier_checks_name_and_arguments():
    task = {"expected_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]}
    ok = asyncio.run(verify_tool_calling(task, '{"tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]}'))
    bad = asyncio.run(verify_tool_calling(task, '{"tool_calls": [{"name": "write_file", "arguments": {"path": "src/app.py"}}]}'))
    assert ok.passed
    assert not bad.passed


def test_execution_pass_rate_respects_tamper_flag():
    res = asyncio.run(verify_execution_pass_rate({"pass_threshold": 0.5}, {"passed_tests": 6, "total_tests": 10, "tamper_flag": False}))
    tampered = asyncio.run(verify_execution_pass_rate({"pass_threshold": 0.5}, {"passed_tests": 10, "total_tests": 10, "tamper_flag": True}))
    assert res.score == 0.6 and res.passed
    assert tampered.score == 0.0 and not tampered.passed
