"""等价性比较辅助：对照执行候选与参考实现并规范化比较结果。

本文件会被原样复制进执行沙箱，只能依赖标准库。
"""

import copy
import math
import types
import builtins


class UncopyableArgumentError(Exception):
    """参数无法 deepcopy 时抛出，调用方据此把该测试标记为 test_defect。"""

    def __init__(self, arg_repr: str):
        self.arg_repr = arg_repr
        super().__init__(f"argument is not copyable: {arg_repr}")


def _safe_deepcopy(value):
    try:
        return copy.deepcopy(value)
    except Exception as e:
        raise UncopyableArgumentError(repr(value)) from e


def _is_float_like(value) -> bool:
    return isinstance(value, float)


def normalize_value(value):
    """规范化返回值：生成器 / 迭代器转 list，其余原样返回。"""
    if isinstance(value, (types.GeneratorType,)) or (
        hasattr(value, "__iter__") and hasattr(value, "__next__")
    ):
        return list(value)
    return value


def compare_calls(candidate, reference, *args, **kwargs) -> tuple:
    """分别用 deepcopy 后的参数调用候选与参考，返回规范化结果元组。

    返回 ("return", 值) 或 ("raise", 异常类型全名, str(异常))；
    参数 deepcopy 失败时抛 UncopyableArgumentError。
    """
    candidate_args = _safe_deepcopy(args)
    candidate_kwargs = _safe_deepcopy(kwargs)
    reference_args = _safe_deepcopy(args)
    reference_kwargs = _safe_deepcopy(kwargs)

    try:
        candidate_result = ("return", normalize_value(candidate(*candidate_args, **candidate_kwargs)))
    except Exception as e:
        candidate_result = ("raise", f"{type(e).__module__}.{type(e).__qualname__}", str(e))

    try:
        reference_result = ("return", normalize_value(reference(*reference_args, **reference_kwargs)))
    except Exception as e:
        reference_result = ("raise", f"{type(e).__module__}.{type(e).__qualname__}", str(e))

    return candidate_result, reference_result


def assert_same_outcome(testcase, outcome_a, outcome_b) -> None:
    """比较两个 compare_calls 结果是否一致，浮点按 rel_tol=1e-9 递归比较。"""
    testcase.assertEqual(outcome_a[0], outcome_b[0], f"结果类别不一致: {outcome_a} vs {outcome_b}")
    if outcome_a[0] == "raise":
        testcase.assertEqual(outcome_a[1], outcome_b[1], f"异常类型不一致: {outcome_a} vs {outcome_b}")
        testcase.assertEqual(outcome_a[2], outcome_b[2], f"异常消息不一致: {outcome_a} vs {outcome_b}")
        return

    _assert_values_equal(testcase, outcome_a[1], outcome_b[1], path="return")


def _assert_values_equal(testcase, value_a, value_b, path: str) -> None:
    """递归比较两个值，浮点数与含浮点的容器用 math.isclose(rel_tol=1e-9)。"""
    if _is_float_like(value_a) and _is_float_like(value_b):
        testcase.assertTrue(
            math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-12),
            f"{path}: 浮点值不一致 {value_a} vs {value_b}",
        )
        return

    if isinstance(value_a, (list, tuple)) and isinstance(value_b, (list, tuple)):
        testcase.assertEqual(
            len(value_a), len(value_b), f"{path}: 序列长度不一致 {len(value_a)} vs {len(value_b)}"
        )
        for idx, (item_a, item_b) in enumerate(zip(value_a, value_b)):
            _assert_values_equal(testcase, item_a, item_b, path=f"{path}[{idx}]")
        return

    if isinstance(value_a, dict) and isinstance(value_b, dict):
        testcase.assertEqual(
            set(value_a.keys()), set(value_b.keys()), f"{path}: 字典键不一致"
        )
        for key in value_a:
            _assert_values_equal(testcase, value_a[key], value_b[key], path=f"{path}[{key!r}]")
        return

    if isinstance(value_a, (set, frozenset)) and isinstance(value_b, (set, frozenset)):
        testcase.assertEqual(value_a, value_b, f"{path}: 集合不一致")
        return

    # 候选与参考各自构造的自定义对象没有 __eq__ 时按类名与属性字典递归比较；
    # 参考类名带 reference_ 前缀，比较时去掉
    if (
        _plain_type_name(value_a) == _plain_type_name(value_b)
        and hasattr(value_a, "__dict__")
        and hasattr(value_b, "__dict__")
        and type(value_a).__eq__ is object.__eq__
    ):
        _assert_values_equal(testcase, vars(value_a), vars(value_b), path=f"{path}.__dict__")
        return

    try:
        testcase.assertEqual(value_a, value_b)
    except AssertionError:
        testcase.assertEqual(
            repr(value_a).replace("reference_", ""),
            repr(value_b).replace("reference_", ""),
            f"{path}: 值不一致",
        )


def _plain_type_name(value) -> str:
    name = type(value).__name__
    return name[len("reference_"):] if name.startswith("reference_") else name
