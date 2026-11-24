"""Unit tests for data preparation (filtering, exact/fuzzy dedup, parquet schema) and YAML configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from math_reasoning_rl.data_prep import (
    convert_open_source_record,
    deduplicate_and_filter,
    is_filtered_problem_type,
    to_verl_parquet_row,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_problem_type_filtering():
    assert is_filtered_problem_type({"type": "选择", "question": "下列选项正确的是"}) is True
    assert is_filtered_problem_type({"type": "判断", "question": "1+1=3"}) is True
    assert is_filtered_problem_type({"type": "证明", "question": "求证三角形全等"}) is True
    assert is_filtered_problem_type({"type": "填空", "question": "A. 1 B. 2 C. 3 D. 4"}) is True
    assert is_filtered_problem_type({"type": "解答", "question": "求求证的值？请证明：a=b"}) is True
    assert is_filtered_problem_type({"type": "计算", "question": "计算 3/4 + 5/6"}) is False


def test_exact_and_fuzzy_dedup_and_parquet_schema():
    raw = [
        {"qid": "1", "type": "计算", "question": "已知方程 x^2 - 5x + 6 = 0，求两根平方和。", "answer": "13"},
        {"qid": "2", "type": "计算", "question": "已知方程 x^2 - 5x + 6 = 0 求两根平方和", "answer": "13"},  # exact after norm
        {"qid": "3", "type": "计算", "question": "已知方程 x^2 - 5x + 6 = 0，求两根的平方和是多少。", "answer": "13"},  # fuzzy
        {
            "qid": "4",
            "type": "解答",
            "question": "已知二次函数 f(x)=x^2-4x+1。(1)求对称轴；(2)求最小值。",
            "subq": [{"id": 1, "answer": "x=2"}, {"id": 2, "answer": "-3", "key_step": "配方"}],
        },
        {"qid": "5", "type": "选择", "question": "A.1 B.2", "answer": "A"},
    ]
    unified = [convert_open_source_record(r, i) for i, r in enumerate(raw)]
    cleaned, stats = deduplicate_and_filter(unified, jaccard_threshold=0.65, ngram_n=3)

    assert stats.filtered_by_type == 1
    assert stats.exact_duplicates == 1
    assert stats.fuzzy_duplicates == 1
    assert stats.kept == 2
    assert stats.rule_routed == 1
    assert stats.judge_routed == 1

    row_rule = to_verl_parquet_row(cleaned[0], index=0)
    row_judge = to_verl_parquet_row(cleaned[1], index=1)
    assert row_rule["extra_info"]["route"] == "rule"
    assert row_judge["extra_info"]["route"] == "judge"
    assert isinstance(row_judge["extra_info"]["subq_json"], str)
    assert len(json.loads(row_judge["extra_info"]["subq_json"])) == 2


def test_configs_match_spec_defaults():
    s1 = yaml.safe_load((PROJECT_ROOT / "configs" / "dapo_8xa800_stage1_8k.yaml").read_text(encoding="utf-8"))
    s2 = yaml.safe_load((PROJECT_ROOT / "configs" / "dapo_8xa800_stage2_16k.yaml").read_text(encoding="utf-8"))
    ref = yaml.safe_load((PROJECT_ROOT / "configs" / "ref_scale_14b_24xh800.yaml").read_text(encoding="utf-8"))

    assert s1["trainer"]["n_gpus_per_node"] == 6
    assert s1["data"]["train_batch_size"] == 64
    assert s1["actor_rollout_ref"]["rollout"]["n"] == 8
    assert s1["actor_rollout_ref"]["rollout"]["response_length"] == 8192
    assert s2["actor_rollout_ref"]["rollout"]["response_length"] == 16384
    assert s1["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "seq-mean-token-sum-norm"
    assert s1["actor_rollout_ref"]["actor"]["loss_scale_factor"] == 8192
    assert s2["actor_rollout_ref"]["actor"]["loss_scale_factor"] == 16384
    assert s1["actor_rollout_ref"]["actor"]["clip_ratio_low"] == 0.2
    assert s1["actor_rollout_ref"]["actor"]["clip_ratio_high"] == 0.28
    assert s1["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert s1["math_rl"]["entropy_band"] == {"enabled": True, "low": 0.45, "high": 0.65, "coef": 0.005}
    assert s1["math_rl"]["pass_rate_weight"]["center"] == 0.25
    assert s1["math_rl"]["overflow_reuse"]["max_reuse_steps"] == 1
    assert s1["math_rl"]["overflow_reuse"]["is_ratio_clip"] == 2.0

    assert ref["trainer"]["nnodes"] == 2
    assert ref["trainer"]["n_gpus_per_node"] == 8
    assert ref["data"]["train_batch_size"] == 128
