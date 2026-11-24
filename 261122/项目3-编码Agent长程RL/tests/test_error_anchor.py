"""测试 3.4 节：报错规范化与 GiGPO step anchor 提取。"""

from __future__ import annotations

from long_horizon_rl.error_anchor import extract_error_anchor, normalize_error_text


def test_normalize_error_text_replaces_volatile_tokens() -> None:
    raw = "File /tmp/tmpa8f3k/pkg/core.py:118 object at 0x7f3a2c90 missing 'sep' with 42"
    norm = normalize_error_text(raw)
    assert "<tmp>" in norm
    assert ":<line>" in norm
    assert "<addr>" in norm
    assert "<str>" in norm
    assert "<num>" in norm
    assert "0x7f3a2c90" not in norm
    assert "'sep'" not in norm


def test_extract_same_anchor_across_different_paths_and_values() -> None:
    tb1 = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/tmp111/pkg/core.py", line 42, in parse\n'
        "    return row['sep']\n"
        "KeyError: 'sep'\n"
    )
    tb2 = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/tmp999/pkg/core.py", line 118, in parse\n'
        "    return row['delim']\n"
        "KeyError: 'delim'\n"
    )
    a1 = extract_error_anchor(tb1, granularity="full")
    a2 = extract_error_anchor(tb2, granularity="full")
    assert a1 is not None
    assert a1 == a2
    assert "KeyError" in a1
    assert "parse()" in a1


def test_extract_assertion_shape_and_first_failure_only() -> None:
    pytest_out = (
        "___________________ test_alpha ___________________\n"
        '  File "/testbed/tests/test_a.py", line 12, in test_alpha\n'
        "    assert result == [1, 2]\n"
        "E   AssertionError: assert [1] == [1, 2]\n"
        "___________________ test_beta ____________________\n"
        '  File "/testbed/tests/test_b.py", line 20, in test_beta\n'
        "    raise ValueError('bad')\n"
        "E   ValueError: bad\n"
    )
    anchor = extract_error_anchor(pytest_out, granularity="full")
    assert anchor is not None
    assert "AssertionError" in anchor
    assert "assert <expr> == <expr>" in anchor
    assert "ValueError" not in anchor


def test_granularity_exception_only_ablation() -> None:
    tb = 'File "/testbed/pkg/core.py", line 10, in parse\nKeyError: "x"\n'
    assert extract_error_anchor(tb, granularity="exception_only") == "KeyError"


def test_non_error_output_returns_none() -> None:
    assert extract_error_anchor("12 passed in 0.43s\n[退出码 0]") is None
    assert extract_error_anchor("") is None


def test_pytest_long_traceback_frame_gives_business_function() -> None:
    from long_horizon_rl.error_anchor import extract_error_anchor

    output = (
        "____ test_parse ____\n"
        "    def test_parse():\n"
        ">       assert parse('a,b') == ['a', 'b']\n"
        "tests/test_core.py:5: \n"
        "pkg/core.py:12: in parse\n"
        "    return text.split(opts['sep'])\n"
        "E   KeyError: 'sep'\n"
        "pkg/core.py:12: KeyError\n"
    )
    assert extract_error_anchor(output) == "KeyError: <str> in parse()"


def test_exact_base_exception_names_are_recognized() -> None:
    from long_horizon_rl.error_anchor import extract_error_anchor

    assert extract_error_anchor("Exception: boom") is not None
    assert extract_error_anchor("StopIteration") == "StopIteration"
