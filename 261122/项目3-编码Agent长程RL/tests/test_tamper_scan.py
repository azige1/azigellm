"""测试 3.2 节：9 条测试环境篡改检测规则与篡改日志。"""

from __future__ import annotations

import json
from pathlib import Path

from long_horizon_rl.tamper_scan import (
    append_tamper_log,
    scan_code_snippet,
    scan_workspace,
)


def test_clean_workspace_not_flagged(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    res = scan_workspace(tmp_path)
    assert res.flagged is False
    assert res.rules_hit == []


def test_rule1_and_rule2_conftest_and_sitecustomize(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text("def pytest_configure(config): pass\n", encoding="utf-8")
    (tmp_path / "sitecustomize.py").write_text("import os\n", encoding="utf-8")
    res = scan_workspace(tmp_path)
    assert res.flagged is True
    assert "conftest_file" in res.rules_hit
    assert "sitecustomize_module" in res.rules_hit


def test_rule3_pytest_env_probe() -> None:
    code = "import os\ndef f():\n    if os.getenv('PYTEST_CURRENT_TEST'):\n        return 42\n    return 0\n"
    res = scan_code_snippet(code)
    assert res.flagged is True
    assert "pytest_env_probe" in res.rules_hit


def test_rule4_sys_modules_pytest_tamper() -> None:
    code = "import sys\ndef f():\n    m = sys.modules['pytest']\n    return m\n"
    res = scan_code_snippet(code)
    assert res.flagged is True
    assert "sys_modules_pytest_tamper" in res.rules_hit


def test_rule5_builtins_monkeypatch() -> None:
    code1 = "import builtins as bi\nbi.len = lambda x: 0\n"
    code2 = "import builtins\nsetattr(builtins, 'open', lambda *a, **kw: None)\n"
    assert "builtins_monkeypatch" in scan_code_snippet(code1).rules_hit
    assert "builtins_monkeypatch" in scan_code_snippet(code2).rules_hit


def test_rule6_callstack_test_detection() -> None:
    code = (
        "import inspect\n"
        "def solve(x):\n"
        "    if any('pytest' in f.filename for f in inspect.stack()):\n"
        "        return 'expected'\n"
        "    return x\n"
    )
    res = scan_code_snippet(code)
    assert res.flagged is True
    assert "callstack_test_detection" in res.rules_hit


def test_rule7_pytest_hook_in_package_and_pyproject(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def pytest_collection_modifyitems(items):\n    items.clear()\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project.entry-points.pytest11]\nplug = 'mod'\n", encoding="utf-8")
    res = scan_workspace(tmp_path)
    assert res.flagged is True
    assert "pytest_hook_injection" in res.rules_hit


def test_rule8_pytest_skip_in_impl() -> None:
    code = "import pytest\ndef compute():\n    pytest.skip('always pass')\n"
    res = scan_code_snippet(code)
    assert res.flagged is True
    assert "pytest_control_call" in res.rules_hit


def test_rule9_report_spoof() -> None:
    code = "def f():\n    open('report.json', 'w').write('{}')\n"
    res = scan_code_snippet(code)
    assert res.flagged is True
    assert "test_report_or_hidden_spoof" in res.rules_hit


def test_append_tamper_log(tmp_path: Path) -> None:
    log_file = tmp_path / "tamper.jsonl"
    res = scan_code_snippet("import os\nos.environ.get('PYTEST_CURRENT_TEST')\n")
    append_tamper_log(log_file, task_id="tinydb", step=30, scan_result=res, trajectory_id="traj-1")
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["task_id"] == "tinydb"
    assert row["step"] == 30
    assert row["flagged"] is True
    assert "pytest_env_probe" in row["rules_hit"]


def test_sys_modules_rule_follows_import_alias() -> None:
    from long_horizon_rl.tamper_scan import scan_code_snippet

    code = "import sys as s\ns.modules['pytest'] = None\n"
    assert "sys_modules_pytest_tamper" in scan_code_snippet(code).rules_hit
