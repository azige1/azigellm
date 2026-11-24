import pytest

from multi_teacher_distillation.warmup_sft import (
    allocate_warmup_samples,
    evaluate_domain_overlap,
    filter_verified_teacher_samples,
    render_full_conversation_with_assistant_mask,
    verify_post_warmup_overlap,
)


class _DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        del tokenize, kwargs
        ids = []
        for idx, msg in enumerate(messages):
            role_id = {"system": 100, "user": 101, "assistant": 102, "tool": 103}[msg["role"]]
            ids.append(900 + idx)
            ids.append(role_id)
            ids.extend(ord(ch) for ch in msg["content"])
            ids.append(999)
        if add_generation_prompt:
            ids.extend([900 + len(messages), 102])
        return ids


def _make_overlap_row(domain: str, kl: float, overlap_count: int):
    # top_k = 20
    s_topk = list(range(1, 21))
    t_topk = list(range(1, overlap_count + 1)) + list(range(100, 100 + (20 - overlap_count)))
    return {
        "domain": domain,
        "student_logprobs": [-1.0 + kl, -1.0 + kl, -5.0],
        "teacher_logprobs": [-1.0, -1.0, -1.0],
        "student_topk_ids": [s_topk, s_topk, s_topk],
        "teacher_topk_ids": [t_topk, t_topk, t_topk],
        "response_mask": [1, 1, 0],  # tool token masked out
    }


def test_overlap_check_allocation_and_post_verification():
    pre_rows = [
        _make_overlap_row("repo_impl", kl=0.24, overlap_count=13),       # 0.65 < 0.75 (gap ~0.10 -> ~3000)
        _make_overlap_row("tool_calling", kl=0.17, overlap_count=14),    # 0.70 < 0.75 (gap ~0.05 -> ~1000)
        _make_overlap_row("math_science", kl=0.08, overlap_count=17),    # 0.85 >= 0.75 (pass)
    ]
    pre_report = evaluate_domain_overlap(pre_rows, max_reverse_kl=0.15, min_topk_overlap=0.75, top_k=20)
    assert set(pre_report["warmup_domains"]) == {"repo_impl", "tool_calling"}

    alloc = allocate_warmup_samples(pre_report, total_samples=4000, min_topk_overlap=0.75)
    assert alloc["math_science"] == 0
    assert alloc["repo_impl"] + alloc["tool_calling"] == 4000
    assert alloc["repo_impl"] > alloc["tool_calling"]

    # Check exact 3:1 allocation when overlap = 0.66 and 0.72
    exact_report = {
        "max_reverse_kl": 0.15,
        "domains": {
            "repo_impl": {"needs_warmup": True, "topk_overlap": 0.66, "reverse_kl": 0.24},
            "tool_calling": {"needs_warmup": True, "topk_overlap": 0.72, "reverse_kl": 0.17},
            "math_science": {"needs_warmup": False, "topk_overlap": 0.84, "reverse_kl": 0.08},
        },
    }
    exact_alloc = allocate_warmup_samples(exact_report, total_samples=4000, min_topk_overlap=0.75)
    assert exact_alloc == {"repo_impl": 3000, "tool_calling": 1000, "math_science": 0}

    teacher_pool = [
        {"domain": "repo_impl", "messages": [{"role": "user", "content": "q"}], "passed": False},
        {"domain": "repo_impl", "messages": [{"role": "user", "content": "q"}], "passed": True},
        {"domain": "tool_calling", "messages": [{"role": "user", "content": "q"}], "passed": True},
        {"domain": "math_science", "messages": [{"role": "user", "content": "q"}], "passed": True},
    ]
    selected = filter_verified_teacher_samples(teacher_pool, {"repo_impl": 1, "tool_calling": 1, "math_science": 0})
    assert len(selected) == 2

    post_rows = [
        _make_overlap_row("repo_impl", kl=0.12, overlap_count=16),       # 0.80
        _make_overlap_row("tool_calling", kl=0.10, overlap_count=16),    # 0.80
        _make_overlap_row("math_science", kl=0.08, overlap_count=17),    # 0.85 (drop 0.0 <= 0.02)
    ]
    check = verify_post_warmup_overlap(pre_report, post_rows)
    assert check["passed"] is True

    bad_post = [
        _make_overlap_row("repo_impl", kl=0.12, overlap_count=16),
        _make_overlap_row("tool_calling", kl=0.10, overlap_count=16),
        _make_overlap_row("math_science", kl=0.08, overlap_count=15),    # 0.75 (drop 0.10 > 0.02)
    ]
    bad_check = verify_post_warmup_overlap(pre_report, bad_post)
    assert bad_check["passed"] is False
    assert bad_check["drift_violations"] == ["math_science"]


def test_full_conversation_rendering_masks_only_assistant_tokens():
    tok = _DummyTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "res"},
        {"role": "assistant", "content": "done"},
    ]
    full_ids, loss_mask = render_full_conversation_with_assistant_mask(tok, messages)
    assert len(full_ids) == len(loss_mask)
    masked_tokens = [tok_id for tok_id, m in zip(full_ids, loss_mask, strict=True) if m == 1]
    assert masked_tokens == [ord("o"), ord("k"), 999, ord("d"), ord("o"), ord("n"), ord("e"), 999]
