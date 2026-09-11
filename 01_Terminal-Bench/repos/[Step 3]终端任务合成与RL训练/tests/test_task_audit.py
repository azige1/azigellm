"""task_audit 包的单元测试。

只覆盖纯静态与解题结果检查；容器级 check_task 需要 Docker，
不在本文件范围内（batch_anomalies 的聚合逻辑例外，使用构造数据）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_audit import Decision, TaskAuditResult, worst_decision
from task_audit import static_checks, container_checks, solution_checks


# ---------------------------------------------------------------------------
# 构造最小任务目录
# ---------------------------------------------------------------------------

GOOD_FINAL_TEST = '''import subprocess
from pathlib import Path


def test_script_runs():
    result = subprocess.run(["python3", "/home/user/etl/run.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"run.py failed: {result.stderr}"


def test_output_content():
    content = Path("/home/user/etl/out.csv").read_text()
    assert "timestamp" in content
    assert len(content.splitlines()) == 11
'''

PASSIVE_FINAL_TEST = '''import os
from pathlib import Path


def test_dir_exists():
    assert os.path.isdir("/home/user/etl")


def test_file_exists():
    assert Path("/home/user/etl/run.py").exists()
'''


def _make_task(root: Path, name: str, final_test: str = GOOD_FINAL_TEST,
               instruction: str = "The nightly etl run.py job drops rows from out.csv, please fix the timestamp filtering.") -> Path:
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()
    (d / "instruction.md").write_text(instruction)
    (d / "task.toml").write_text(
        'version = "0.1"\n\n[metadata]\nauthor_name = "t"\ndifficulty = "medium"\ncategory = "data_processing"\ntags = ["data_processing"]\n'
    )
    (d / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (d / "environment" / "task.json").write_text(json.dumps({"description": instruction, "truth": "t", "name": name}))
    (d / "environment" / "test_initial_state.py").write_text("def test_ok():\n    assert True\n")
    (d / "tests" / "test.sh").write_text("#!/bin/bash\npytest /tests/test_final_state.py\n")
    (d / "tests" / "test_final_state.py").write_text(final_test)
    return d


# ---------------------------------------------------------------------------
# static_checks
# ---------------------------------------------------------------------------

class TestStaticChecks:
    def test_good_task_passes(self, tmp_path):
        task = _make_task(tmp_path, "task_000001_aa")
        result = static_checks.check_task(task)
        assert result.decision == Decision.PASS
        assert result.metrics["structural_score"] == 1.0
        assert result.metrics["desc_overlap"] > 0
        assert result.metrics["assert_diversity"] >= 2

    def test_missing_files_rejected(self, tmp_path):
        task = tmp_path / "task_000002_bb"
        task.mkdir()
        result = static_checks.check_task(task)
        assert result.decision == Decision.REJECT
        assert "missing_required_files" in result.flags

    def test_passive_verifier_rejected(self, tmp_path):
        task = _make_task(tmp_path, "task_000003_cc", final_test=PASSIVE_FINAL_TEST)
        result = static_checks.check_task(task)
        assert result.decision == Decision.REJECT
        assert "no_outcome_asserts" in result.flags

    def test_syntax_error_rejected(self, tmp_path):
        task = _make_task(tmp_path, "task_000004_dd")
        (task / "tests" / "test_final_state.py").write_text("def broken(:\n")
        result = static_checks.check_task(task)
        assert result.decision == Decision.REJECT
        assert "test_syntax_error" in result.flags

    def test_unknown_difficulty_flagged(self, tmp_path):
        task = _make_task(tmp_path, "task_000005_ee")
        (task / "task.toml").write_text(
            'version = "0.1"\n\n[metadata]\ndifficulty = "impossible"\ncategory = "x"\n'
        )
        result = static_checks.check_task(task)
        assert result.decision == Decision.FLAG
        assert "metadata_unknown_difficulty" in result.flags

    def test_keyword_extraction_excludes_stopwords(self):
        kws = static_checks._extract_keywords("The task is about the server and the logs")
        assert "the" not in kws
        assert "server" in kws

    def test_dataset_summary_counts(self, tmp_path):
        _make_task(tmp_path, "task_000010_aa")
        _make_task(tmp_path, "task_000011_bb", final_test=PASSIVE_FINAL_TEST)
        results = static_checks.check_batch(sorted(tmp_path.iterdir()))
        summary = static_checks.dataset_summary(results)
        assert summary["total"] == 2
        assert summary["pass"] == 1
        assert summary["reject"] == 1
        assert 0.0 < summary["mean_structural_score"] < 1.0


# ---------------------------------------------------------------------------
# worst_decision
# ---------------------------------------------------------------------------

class TestWorstDecision:
    def _r(self, decision: Decision) -> TaskAuditResult:
        return TaskAuditResult("t", Path("."), decision)

    def test_reject_dominates(self):
        assert worst_decision([self._r(Decision.PASS), self._r(Decision.REJECT)]) == Decision.REJECT

    def test_flag_over_pass(self):
        assert worst_decision([self._r(Decision.PASS), self._r(Decision.FLAG)]) == Decision.FLAG

    def test_all_pass(self):
        assert worst_decision([self._r(Decision.PASS)]) == Decision.PASS


# ---------------------------------------------------------------------------
# container_checks.batch_anomalies（不需要 Docker 的聚合逻辑）
# ---------------------------------------------------------------------------

class TestBatchAnomalies:
    def _r(self, flags: list[str]) -> TaskAuditResult:
        decision = Decision.REJECT if any(f in ("build_or_initial_test_failed", "dockerfile_parse_error", "trivially_solved") for f in flags) else Decision.FLAG
        return TaskAuditResult("t", Path("."), decision, {}, flags)

    def test_healthy_batch(self):
        results = [TaskAuditResult("t", Path("."), Decision.PASS) for _ in range(10)]
        anomalies = container_checks.batch_anomalies(results)
        assert anomalies["warnings"] == []
        assert anomalies["build_failure_rate"] == 0.0

    def test_high_build_failure_warns(self):
        results = [self._r(["build_or_initial_test_failed"]) for _ in range(5)] + \
                  [TaskAuditResult("t", Path("."), Decision.PASS) for _ in range(5)]
        anomalies = container_checks.batch_anomalies(results)
        assert "high_build_failure_rate" in anomalies["warnings"]

    def test_empty_batch(self):
        anomalies = container_checks.batch_anomalies([])
        assert anomalies["total"] == 0


# ---------------------------------------------------------------------------
# solution_checks
# ---------------------------------------------------------------------------

class TestSolutionChecks:
    def test_unevaluated_task_flagged(self, tmp_path):
        task = _make_task(tmp_path, "task_000020_aa")
        result = solution_checks.check_task(task)
        assert result.decision == Decision.FLAG
        assert "unevaluated" in result.flags

    def test_never_solved_good_verifier_flagged(self, tmp_path):
        task = _make_task(tmp_path, "task_000021_bb")
        (task / "solution" / "solution.json").write_text(json.dumps({
            "num_runs": 8, "num_success": 0, "pass_at_k": {"1": 0.0},
        }))
        result = solution_checks.check_task(task)
        assert result.decision == Decision.FLAG
        assert "never_solved" in result.flags

    def test_never_solved_bad_verifier_rejected(self, tmp_path):
        task = _make_task(tmp_path, "task_000022_cc", final_test=PASSIVE_FINAL_TEST)
        (task / "solution" / "solution.json").write_text(json.dumps({
            "num_runs": 8, "num_success": 0, "pass_at_k": {"1": 0.0},
        }))
        result = solution_checks.check_task(task)
        assert result.decision == Decision.REJECT
        assert "verifier_likely_wrong" in result.flags

    def test_solved_task_passes(self, tmp_path):
        task = _make_task(tmp_path, "task_000023_dd")
        (task / "solution" / "solution.json").write_text(json.dumps({
            "num_runs": 8, "num_success": 3, "pass_at_k": {"1": 0.375},
        }))
        result = solution_checks.check_task(task)
        assert result.decision == Decision.PASS

    def test_check_run_high_error_rate(self, tmp_path):
        result_json = tmp_path / "result.json"
        result_json.write_text(json.dumps({
            "n_total_trials": 100,
            "stats": {
                "n_errors": 10,
                "evals": {"agent__model__tasks": {"n_trials": 100, "metrics": [{"mean": 0.5}]}},
            },
        }))
        out = solution_checks.check_run(result_json)
        assert out["ok"] is True
        assert "high_error_rate" in out["warnings"]

    def test_check_run_low_pass_rate(self, tmp_path):
        result_json = tmp_path / "result.json"
        result_json.write_text(json.dumps({
            "n_total_trials": 100,
            "stats": {
                "n_errors": 1,
                "evals": {"agent__model__tasks": {"n_trials": 100, "metrics": [{"mean": 0.05}]}},
            },
        }))
        out = solution_checks.check_run(result_json)
        assert "low_overall_pass_rate" in out["warnings"]

    def test_check_run_missing_file(self, tmp_path):
        out = solution_checks.check_run(tmp_path / "nope.json")
        assert out["ok"] is False
