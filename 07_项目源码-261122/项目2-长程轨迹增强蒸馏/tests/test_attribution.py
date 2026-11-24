"""归因测试：九类规则各一个正例与反例；统计拒绝未声明的子集。"""

import pytest

from trajectory_tools.attribution import (
    RULES,
    AttributionContext,
    deepest_src_line,
    extract_failures,
    summarize,
)


def _assistant(content="", tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


def _bash_call(idx_cmd_pairs):
    """构造 assistant 消息：每个元素是 (bash 命令, [工具结果...])。"""
    messages = []
    call_no = 0
    for command, outputs in idx_cmd_pairs:
        call_no += 1
        call_id = f"c{call_no}"
        messages.append(
            _assistant(
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": %s}' % repr(command)},
                    }
                ]
            )
        )
        for out, is_error in outputs:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": out, "is_error": is_error})
    return messages


def _write_call(path):
    return {
        "id": "w1",
        "type": "function",
        "function": {"name": "write_file", "arguments": '{"path": "%s", "content": "x"}' % path},
    }


def make_ctx(messages, **overrides):
    """构造归因上下文；meta 可用 overrides["meta"] 覆盖。"""
    for idx, m in enumerate(messages):
        m.setdefault("g", idx)
    traj = {"messages": messages, "meta": overrides.pop("meta", {})}
    ctx = AttributionContext(
        traj=traj,
        task={"instance_id": "tinydb", "repo": "msiemens/tinydb", "src_dir": "tinydb"},
        failures=overrides.pop("failures", []),
        collection_error=overrides.pop("collection_error", False),
        bash_commands=overrides.pop("bash_commands", None)
        or [
            __import__("json").loads(c["function"]["arguments"]).get("command", "")
            for m in messages
            for c in (m.get("tool_calls") or [])
            if c["function"]["name"] == "bash"
        ],
        tool_outputs=[m.get("content", "") for m in messages if m.get("role") == "tool"],
        tool_error_flags=[bool(m.get("is_error")) for m in messages if m.get("role") == "tool"],
        pass_rate=overrides.pop("pass_rate", 0.0),
        max_blind=overrides.pop("max_blind", 0),
        blind_segments=overrides.pop("blind_segments", 0),
    )
    return ctx


RULES_BY_ID = {rule_id: match for rule_id, _, _, match in RULES}


def test_premature_completion():
    # 完成宣告后最后一次改动之后没有验证 → 命中
    ctx = make_ctx([_assistant("TASK COMPLETE", [_write_call("a.py")])])
    assert RULES_BY_ID["premature_completion"](ctx)
    # 有验证且在最后改动之后 → 不命中
    ctx = make_ctx(
        _bash_call([("pytest tests", [("passed", False)])]) + [_assistant("TASK COMPLETE", [_write_call("a.py")])]
        + _bash_call([("pytest tests", [("passed", False)])])
    )
    assert not RULES_BY_ID["premature_completion"](ctx)


def test_incomplete():
    ctx = make_ctx([_assistant("working")], meta={"result_subtype": "error_max_turns"})
    assert RULES_BY_ID["incomplete"](ctx)
    ctx = make_ctx([_assistant("TASK COMPLETE")], meta={"result_subtype": "success"})
    assert not RULES_BY_ID["incomplete"](ctx)


def test_context_truncation():
    messages = [
        {"role": "user", "content": "This session is being continued from a previous conversation."},
        _assistant(tool_calls=[{"id": "e1", "type": "function", "function": {"name": "edit_file", "arguments": "{}"}}]),
        {"role": "tool", "tool_call_id": "e1", "content": "old_string 未匹配", "is_error": True},
    ]
    ctx = make_ctx(messages, meta={"has_compact_boundary": True})
    assert RULES_BY_ID["context_truncation"](ctx)
    ctx = make_ctx(messages, meta={"has_compact_boundary": False})
    assert not RULES_BY_ID["context_truncation"](ctx)


def test_navigation_loop():
    nav = [("ls", [("a.py", False)])] * 8
    ctx = make_ctx(_bash_call(nav))
    assert RULES_BY_ID["navigation_loop"](ctx)
    ctx = make_ctx(_bash_call(nav[:3]))
    assert not RULES_BY_ID["navigation_loop"](ctx)


def test_blind_editing():
    ctx = make_ctx([_assistant()], max_blind=5)
    assert RULES_BY_ID["blind_editing"](ctx)
    ctx = make_ctx([_assistant()], max_blind=2)
    assert not RULES_BY_ID["blind_editing"](ctx)


def test_packaging_failure():
    from trajectory_tools.attribution import Failure

    ctx = make_ctx([_assistant()], collection_error=True)
    assert RULES_BY_ID["packaging_failure"](ctx)
    ctx = make_ctx(
        [_assistant()],
        failures=[Failure("t::x", "ModuleNotFoundError", None, "")],
    )
    assert RULES_BY_ID["packaging_failure"](ctx)
    ctx = make_ctx([_assistant()], failures=[])
    assert not RULES_BY_ID["packaging_failure"](ctx)


def test_verification_mismatch():
    ctx = make_ctx(_bash_call([("pytest tests", [("1 passed [退出码 0]", False)])]), pass_rate=0.2)
    assert RULES_BY_ID["verification_mismatch"](ctx)
    ctx = make_ctx(_bash_call([("pytest tests", [("1 passed [退出码 0]", False)])]), pass_rate=0.9)
    assert not RULES_BY_ID["verification_mismatch"](ctx)


def test_resource_exhaustion():
    ctx = make_ctx(_bash_call([("pytest tests", [("MemoryError", True)])]))
    assert RULES_BY_ID["resource_exhaustion"](ctx)
    ctx = make_ctx(_bash_call([("pytest tests", [("1 passed", False)])]))
    assert not RULES_BY_ID["resource_exhaustion"](ctx)


def test_answer_access():
    ctx = make_ctx(_bash_call([("pip download tinydb", [("", False)])]))
    assert RULES_BY_ID["answer_access"](ctx)
    ctx = make_ctx(_bash_call([("cat hidden_tests/test_a.py", [("", False)])]))
    assert RULES_BY_ID["answer_access"](ctx)
    ctx = make_ctx(_bash_call([("pytest tests", [("1 passed", False)])]))
    assert not RULES_BY_ID["answer_access"](ctx)
    # 安装目标库本身算答案获取，安装其他依赖与以可编辑方式安装工作区不算
    ctx = make_ctx(_bash_call([("pip install tinydb==4.8.0", [("", False)])]))
    assert RULES_BY_ID["answer_access"](ctx)
    ctx = make_ctx(_bash_call([("pip install attrs pyyaml", [("", False)])]))
    assert not RULES_BY_ID["answer_access"](ctx)
    ctx = make_ctx(_bash_call([("pip install -e .", [("", False)])]))
    assert not RULES_BY_ID["answer_access"](ctx)
    ctx = make_ctx(_bash_call([("pip install -r requirements.txt", [("", False)])]))
    assert not RULES_BY_ID["answer_access"](ctx)


def test_deepest_src_line():
    repr_text = 'File "tinydb/core.py", line 12, in insert\nFile "tests/test_x.py", line 3'
    line = deepest_src_line(repr_text, "tinydb")
    assert line == "tinydb/core.py:12"


def test_extract_failures_missing_report(tmp_path):
    failures, collection_error = extract_failures(tmp_path / "nope.json", "", "tinydb")
    assert failures == [] and collection_error


def test_summarize_rejects_undeclared_subset():
    results = {"a": ["blind_editing"]}
    index = [{"id": "a", "hidden_pass_rate": 0.5}]
    with pytest.raises(ValueError):
        summarize(results, index, excluded=1)
    table = summarize(results, index, condition="hidden_pass_rate < 0.9", excluded=1)
    assert table[0]["excluded"] == 1 and table[0]["condition"] == "hidden_pass_rate < 0.9"
