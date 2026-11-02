"""行为指标测试：盲改段数、最长盲改、签名错位率、工具报错率。"""

import subprocess
from pathlib import Path

from trajectory_tools.behavior_metrics import (
    compute_blind_metrics,
    is_verification_command,
    signature_mismatch_rate,
    tool_error_rate,
)


def _bash(command):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{abs(hash(command)) % 10**6}",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": %s}' % repr(command)},
            }
        ],
    }


def _write(path):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"w{abs(hash(path)) % 10**6}",
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path": %s, "content": "x"}' % repr(path)},
            }
        ],
    }


def test_is_verification_command():
    assert is_verification_command("pytest tests/")
    assert is_verification_command("python -m pytest tests/test_a.py")
    assert is_verification_command('python -c "import tinydb"')
    assert is_verification_command("python tests/test_a.py")
    assert not is_verification_command("ls -la")
    assert not is_verification_command("cat tinydb/core.py")


def test_blind_metrics():
    # 3 次写 → pytest → 5 次写：max_blind=5，段数 2
    messages = []
    messages += [_write(f"f{i}.py") for i in range(3)]
    messages.append(_bash("pytest tests"))
    messages += [_write(f"g{i}.py") for i in range(5)]
    result = compute_blind_metrics(messages)
    assert result == {"max_blind": 5, "blind_segments": 2}
    # 两次写之间夹一次 pytest：max_blind=2，无段
    messages = [_write("a.py"), _write("b.py"), _bash("pytest tests"), _write("c.py"), _write("d.py")]
    result = compute_blind_metrics(messages)
    assert result == {"max_blind": 2, "blind_segments": 0}


def test_tool_error_rate():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok", "is_error": False},
        {"role": "tool", "tool_call_id": "c2", "content": "boom", "is_error": True},
    ]
    result = tool_error_rate(messages)
    assert result == {"tool_calls": 2, "tool_errors": 1, "error_rate": 0.5}


def test_signature_mismatch_rate(tmp_path):
    """在临时 git 仓库上验证签名错位率：改签名、删函数、新增函数各计一次。"""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    src = repo_dir / "pkg"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "core.py").write_text(
        "def keep(a, b=1):\n    return a\n\n"
        "def change(a):\n    return a\n\n"
        "def remove(a):\n    return a\n\n"
        "class K:\n    def meth(self, x):\n        return x\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], cwd=repo_dir, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    final_dir = tmp_path / "final"
    (final_dir / "pkg").mkdir(parents=True)
    (final_dir / "pkg" / "core.py").write_text(
        "def keep(a, b=1):\n    return a\n\n"
        "def change(a, extra=None):\n    return a\n\n"
        "def added(a):\n    return a\n",
        encoding="utf-8",
    )
    result = signature_mismatch_rate(repo_dir, final_dir, "pkg", base_commit)
    # change 改签名 + remove 被删 + K.meth 被删 = 3 处错位，共 4 个签名
    assert result["total_signatures"] == 4
    assert result["mismatched"] == 3
    assert result["mismatch_rate"] == 0.75
