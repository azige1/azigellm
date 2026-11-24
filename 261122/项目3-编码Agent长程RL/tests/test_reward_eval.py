"""测试 3.2 节：判分前清理模型测试产物与通过比例 reward 计算。"""

from __future__ import annotations

from pathlib import Path

from long_horizon_rl.reward_eval import (
    compute_reward_from_counts,
    scrub_model_test_artifacts,
    strip_pyproject_pytest_sections,
    strip_setup_cfg_pytest_sections,
)
from long_horizon_rl.tamper_scan import TamperScanResult


def test_strip_setup_cfg_keeps_metadata_and_removes_pytest() -> None:
    raw = (
        "[metadata]\nname = mypkg\n\n"
        "[tool:pytest]\naddopts = -k nothing\n\n"
        "[pytest]\ntestpaths = fake\n"
    )
    cleaned = strip_setup_cfg_pytest_sections(raw)
    assert "[metadata]" in cleaned
    assert "name = mypkg" in cleaned
    assert "tool:pytest" not in cleaned
    assert "[pytest]" not in cleaned


def test_strip_pyproject_removes_only_pytest_section() -> None:
    raw = (
        "[project]\nname = \"demo\"\n\n"
        "[tool.pytest.ini_options]\naddopts = \"-q\"\n\n"
        "[tool.ruff]\nline-length = 100\n"
    )
    cleaned = strip_pyproject_pytest_sections(raw)
    assert "[project]" in cleaned
    assert "[tool.ruff]" in cleaned
    assert "tool.pytest" not in cleaned


def test_scrub_model_test_artifacts(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_fake.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text("[tox]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    (tmp_path / "setup.cfg").write_text("[metadata]\nname=x\n[tool:pytest]\naddopts=-x\n", encoding="utf-8")
    (tmp_path / "pkg.py").write_text("x = 1\n", encoding="utf-8")

    removed = scrub_model_test_artifacts(tmp_path, test_dir="tests")
    assert "tests" in removed
    assert "pytest.ini" in removed
    assert "tox.ini" in removed
    assert "conftest.py" in removed
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "pytest.ini").exists()
    assert (tmp_path / "pkg.py").exists()
    assert "tool:pytest" not in (tmp_path / "setup.cfg").read_text(encoding="utf-8")


def test_compute_reward_modes_and_tamper_zeroing() -> None:
    r_ratio, p_ratio = compute_reward_from_counts(29, 47, mode="pass_ratio")
    assert abs(r_ratio - 29 / 47) < 1e-9
    assert abs(p_ratio - 29 / 47) < 1e-9

    r_bin_partial, _ = compute_reward_from_counts(29, 47, mode="binary")
    r_bin_full, _ = compute_reward_from_counts(47, 47, mode="binary")
    assert r_bin_partial == 0.0
    assert r_bin_full == 1.0

    flagged = TamperScanResult(flagged=True, rules_hit=["conftest_file"])
    r_tamper, p_tamper = compute_reward_from_counts(47, 47, mode="pass_ratio", tamper_result=flagged)
    assert r_tamper == 0.0
    assert p_tamper == 1.0


def test_tampered_workspace_scores_zero_without_running_tests(tmp_path: Path) -> None:
    from long_horizon_rl.reward_eval import evaluate_repo_workspace

    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "conftest.py").write_text("import pytest\n", encoding="utf-8")
    # 未提供判分后端所需字段：若篡改检测后仍继续执行测试，这里会因缺字段报错
    outcome = evaluate_repo_workspace(tmp_path, {"instance_id": "demo"}, scoring_backend="host")
    assert outcome.reward == 0.0
    assert outcome.tamper_flag is True
    assert outcome.infra_failure is False
    assert "conftest_file" in outcome.rules_hit


def test_unparseable_setup_cfg_scores_zero(tmp_path: Path) -> None:
    from long_horizon_rl.reward_eval import evaluate_repo_workspace

    (tmp_path / "setup.cfg").write_text("no section header\n[pytest\n", encoding="utf-8")
    outcome = evaluate_repo_workspace(tmp_path, {"instance_id": "demo"}, scoring_backend="host")
    assert outcome.reward == 0.0
    assert outcome.infra_failure is False
