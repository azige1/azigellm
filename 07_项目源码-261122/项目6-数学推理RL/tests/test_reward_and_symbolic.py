"""Unit tests for 6.1: gates, SymPy symbolic verification, per-subquestion judge routing, and FPR stats."""

from __future__ import annotations

import json

import httpx

from math_reasoning_rl.config import RewardRoutingConfig
from math_reasoning_rl.false_positive_eval import compute_calibration_metrics
from math_reasoning_rl.judge_client import OpenAICompatJudgeClient, parse_judge_response
from math_reasoning_rl.reward_scorer import (
    check_response_gate,
    compute_ngram_repetition_rate,
    compute_score_sync,
    split_thinking_and_answer,
)
from math_reasoning_rl.symbolic_verify import are_expressions_equivalent, extract_rule_answer, verify_rule_answer


def test_split_thinking_and_answer_and_gates():
    thinking, answer, ok = split_thinking_and_answer("<think>先计算 2+3=5</think>答案是 \\boxed{5}")
    assert ok is True
    assert thinking == "先计算 2+3=5"
    assert answer == "答案是 \\boxed{5}"

    # Unbalanced tags fail the gate.
    passed, reason, *_ = check_response_gate("<think>没有闭合标签")
    assert passed is False
    assert reason == "unbalanced_think_tags"

    # High 4-gram repetition in thinking segment triggers -1 gate even if answer is clean.
    rep_think = " ".join(["一步步计算结果"] * 20)
    assert compute_ngram_repetition_rate(rep_think, n=4) > 0.3
    passed, reason, *_ = check_response_gate(f"<think>{rep_think}</think>\\boxed{{5}}")
    assert passed is False
    assert reason == "thinking_repetition_exceeded"

    # High 4-gram repetition in answer segment triggers -1 gate.
    rep_ans = " ".join(["重复结论片段"] * 20)
    passed, reason, *_ = check_response_gate(f"<think>正常推导过程 x=5</think>{rep_ans}")
    assert passed is False
    assert reason == "answer_repetition_exceeded"


def test_symbolic_verification_equivalence():
    assert extract_rule_answer("推导后得 \\boxed{\\frac{5}{4}}") == r"\frac{5}{4}"
    assert are_expressions_equivalent(r"\frac{10}{8}", r"\frac{5}{4}")
    assert are_expressions_equivalent("1.25", r"\frac{5}{4}")
    assert are_expressions_equivalent(r"2\sqrt{3} + 3\sqrt{3}", r"5\sqrt{3}")
    assert are_expressions_equivalent("x = 2", "2")
    assert not are_expressions_equivalent(r"\frac{3}{4}", r"\frac{5}{4}")

    ok, extracted = verify_rule_answer("最终结果为 \\boxed{\\dfrac{5}{4}}", r"\frac{5}{4}")
    assert ok is True
    assert extracted == r"\dfrac{5}{4}"


def test_rule_routed_compute_score():
    resp_ok = "<think>通分相加得到 9/12 + 10/12 - 4/12 = 15/12 = 5/4。</think>最终答案是 \\boxed{\\frac{15}{12}}。"
    out_ok = compute_score_sync(
        data_source="opensource_math",
        solution_str=resp_ok,
        ground_truth=r"\frac{5}{4}",
        extra_info={"qid": "q1", "type": "计算", "route": "rule", "subq": [{"id": 1, "answer": r"\frac{5}{4}"}]},
    )
    assert out_ok["score"] == 1.0
    assert out_ok["acc"] == 1.0
    assert out_ok["route"] == "rule"

    resp_bad_gate = "<think>没有结束标签"
    out_bad = compute_score_sync(
        data_source="opensource_math",
        solution_str=resp_bad_gate,
        ground_truth=r"\frac{5}{4}",
        extra_info={"qid": "q1", "type": "计算", "route": "rule"},
    )
    assert out_bad["score"] == -1.0
    assert out_bad["gate_passed"] == 0.0


def test_judge_routed_all_subquestions_required():
    call_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = body["messages"][0]["content"]
        call_log.append(prompt)
        if "小问编号：1" in prompt:
            content = json.dumps(
                {"extracted_answer": "x=2", "reason": "对称轴正确", "verdict": 1},
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {"extracted_answer": "未作答", "reason": "第二问缺失", "verdict": 0},
                ensure_ascii=False,
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(handler)
    cfg = RewardRoutingConfig(judge_base_url="http://mock-judge/v1", judge_model="Qwen/Qwen3-32B")
    client = OpenAICompatJudgeClient(cfg, transport=transport)

    subq = [
        {"id": 1, "answer": "x=2"},
        {"id": 2, "answer": "最小值为 -3", "key_step": "配方"},
    ]
    response = "<think>对称轴公式 -b/(2a)=2。</think>(1) 对称轴方程为 x=2；(2) 略。"
    res = compute_score_sync(
        data_source="k12_self",
        solution_str=response,
        ground_truth=json.dumps(subq, ensure_ascii=False),
        extra_info={
            "qid": "k12-alg-30417",
            "type": "解答",
            "route": "judge",
            "question": "已知 f(x)=x^2-4x+1...",
            "subq_json": json.dumps(subq, ensure_ascii=False),
        },
        judge_client=client,
        config=cfg,
    )
    assert len(call_log) == 2
    assert res["route"] == "judge"
    assert json.loads(res["sub_verdict"]) == [1, 0]
    assert res["score"] == -1.0

    # Test parse_judge_response with thinking wrapper and markdown code block.
    parsed = parse_judge_response(
        '<think>分析</think>```json\n{"extracted_answer": "最小值为-3", "reason": "ok", "verdict": 1}\n```',
        subq_id=2,
    )
    assert parsed.verdict == 1
    assert parsed.extracted_answer == "最小值为-3"


def test_false_positive_calibration_metrics():
    records = [
        {"gold_verdict": 1},
        {"gold_verdict": 0},
        {"gold_verdict": 0},
        {"gold_verdict": 0},
        {"gold_verdict": 1},
    ]
    preds = [1, 1, 0, 0, 0]  # 1 TP, 1 FP, 2 TN, 1 FN
    m = compute_calibration_metrics(records, preds, strategy="subq_extract_then_judge")
    assert m.tp == 1
    assert m.fp == 1
    assert m.tn == 2
    assert m.fn == 1
    assert abs(m.fpr_over_total - 0.2) < 1e-9
    assert abs(m.fpr_conditional - (1 / 3)) < 1e-9


def test_symbolic_edge_cases_from_review():
    from math_reasoning_rl.symbolic_verify import are_expressions_equivalent, extract_last_boxed

    # 方程不能只比较右侧
    assert not are_expressions_equivalent("x+1=2", "2")
    # 单变量赋值可以省略左侧
    assert are_expressions_equivalent("x=2", "2")
    # 常数倍的等价方程
    assert are_expressions_equivalent("2*x=4", "x=2")
    # 无穷与有限常数不等价
    assert not are_expressions_equivalent("oo", "5")
    # 容器答案不抛异常
    assert not are_expressions_equivalent("(1,2)", "3")
    # 两种框选宏取位置最后的一个
    assert extract_last_boxed(r"先得 \boxed{1}，修正后 \fbox{2}") == "2"
