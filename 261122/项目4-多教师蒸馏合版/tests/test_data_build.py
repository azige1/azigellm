import json

from multi_teacher_distillation.data.build import build_prompt_pool
from multi_teacher_distillation.data.decontaminate import decontaminate


def test_build_prompt_pool_maps_ten_candidates_to_eight_teachers():
    inputs = {
        "func_gen": [{"prompt": "写快排"}],
        "competitive_coding": [{"prompt": "最短路"}],
        "repo_impl": [{"instruction": "修复解析器"}],
        "issue_fix": [{"instruction": "修复 Issue #1"}],
        "test_gen": [{"prompt": "写测试"}],
        "math": [{"prompt": "1+1=?", "answer": "2"}],
        "science": [{"prompt": "重力加速度", "answer": "9.8"}],
        "instruction_following": [{"prompt": "30词内"}],
        "tool_calling": [{"prompt": "调用 read_file"}],
        "general_chat": [{"prompt": "解释递归"}],
        "self_anchor": [{"prompt": "总结"}],
    }
    rows, stats = build_prompt_pool(inputs, seed=17)
    assert stats["total"] == 11
    assert stats["counts"]["code_gen"] == 2
    assert stats["counts"]["math_science"] == 2
    assert stats["counts"]["self_anchor"] == 1
    for row in rows:
        assert row["domain"] == row["teacher"]
        assert row["segment_index"] == 0
        assert row["extra_info"]["domain"] == row["domain"]


def test_decontamination_removes_overlapping_eval_items():
    train = [
        {"prompt": "请实现一个支持并发安全的缓存类并包含过期清理逻辑的完整 Python 示例", "entities": ["cache_v1"]},
        {"prompt": "写一个计算斐波那契数列第 n 项的函数", "entities": ["fib"]},
    ]
    eval_rows = [{"prompt": "请实现一个支持并发安全的缓存类并包含过期清理逻辑的完整 Python 示例", "entities": ["cache_v1"]}]
    kept, removed = decontaminate(train, eval_rows)
    assert len(kept) == 1
    assert len(removed) == 1
