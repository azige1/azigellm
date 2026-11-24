"""注入测试：预算公式、完成宣告必选、质量门、写回不改 tool_calls。"""

from trajectory_tools.injection import (
    Trigger,
    allocate_budget,
    apply_reviews,
    choose_trigger_points,
    pass_quality_gate,
)


def test_budget_formula():
    # n=24、7 万字符：base = floor(24*10/34) = 7，>5 万字符 +1 → e=8，间隔 max(2, floor(24/9)) = 2
    budget = allocate_budget(24, 70_000)
    assert budget == {"base": 8, "e": 8, "interval": 2}
    budget = allocate_budget(24, 120_000)
    assert budget["e"] == 10  # +2 后封顶 10
    budget = allocate_budget(3, 10_000)
    assert budget["base"] == 1 and budget["e"] == 1


def test_completion_claim_mandatory():
    triggers = [Trigger(g=2, type="error_to_fix"), Trigger(g=9, type="completion_claim")]
    chosen = choose_trigger_points(triggers, char_len=10_000)
    assert any(t.type == "completion_claim" and t.g == 9 for t in chosen)


def test_priority_and_interval():
    # 候选多于预算时按优先级取，且保持最小间隔
    triggers = [Trigger(g=i, type="plan_update") for i in range(20)]
    triggers.insert(5, Trigger(g=100, type="error_to_fix"))
    chosen = choose_trigger_points(triggers, char_len=10_000)
    e = allocate_budget(20, 10_000)["e"]
    assert len([t for t in chosen if t.type != "completion_claim"]) == e
    types = [t.type for t in chosen]
    assert "error_to_fix" in types


def test_quality_gate():
    good = (
        "I have completed the storage module and verified by running pytest on the new tests. "
        "The remaining work is the query layer, which is still in progress. "
        "Edge cases around empty tables are not yet verified. "
        "I still need to run the full suite once more before finishing."
    )
    assert pass_quality_gate(good, "completion_claim")
    # 词数不足
    assert not pass_quality_gate("Done and verified by tests.", "completion_claim")
    # 完成宣告缺验证词
    no_verification = " ".join(["I completed the module and the remaining work is small."] * 8)
    assert not pass_quality_gate(no_verification, "completion_claim")
    # 含祈使句
    imperative = good + "\nRun pytest again."
    assert not pass_quality_gate(imperative, "error_to_fix")
    # 无状态词
    flat = " ".join(["The code was changed in several places today."] * 12)
    assert not pass_quality_gate(flat, "error_to_fix")


def test_apply_reviews_keeps_tool_calls():
    traj = {
        "messages": [
            {"g": 0, "role": "system", "content": "sys"},
            {
                "g": 1,
                "role": "assistant",
                "content": "done",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"g": 2, "role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "meta": {},
    }
    original_calls = traj["messages"][1]["tool_calls"]
    review = (
        "I have completed the module and verified by running pytest. "
        "The remaining items are minor and still need review. "
        "I still need to double check the edge cases before the final run."
    )
    result = apply_reviews(traj, {1: review})
    assert result is not None
    assert result["messages"][1]["content"].startswith(review)
    assert result["messages"][1]["tool_calls"] == original_calls
    assert result["meta"]["injections"][0]["g"] == 1
    # 原轨迹不被修改
    assert traj["messages"][1]["content"] == "done"
    assert "injections" not in traj["meta"]


def test_apply_reviews_none_returns_none():
    traj = {"messages": [{"g": 0, "role": "assistant", "content": "x"}], "meta": {}}
    assert apply_reviews(traj, {}) is None
