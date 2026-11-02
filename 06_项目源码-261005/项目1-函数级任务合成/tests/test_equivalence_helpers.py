"""equivalence_helpers 的单元测试：浮点容差、异常比较、生成器与 deepcopy 隔离。"""

import threading
import unittest

from codeharvest.execution.equivalence_helpers import (
    UncopyableArgumentError,
    assert_same_outcome,
    compare_calls,
)


def add(a, b):
    return a + b


def raise_value_error(x):
    raise ValueError(f"bad {x}")


def make_generator(n):
    return (i for i in range(n))


class CompareCallsTest(unittest.TestCase):

    def test_return_outcome(self):
        outcome_a, outcome_b = compare_calls(add, add, 1, 2)
        self.assertEqual(outcome_a, ("return", 3))
        self.assertEqual(outcome_a, outcome_b)

    def test_raise_outcome(self):
        outcome_a, outcome_b = compare_calls(raise_value_error, raise_value_error, 5)
        self.assertEqual(outcome_a[0], "raise")
        self.assertIn("ValueError", outcome_a[1])
        self.assertEqual(outcome_a, outcome_b)

    def test_generator_normalized(self):
        outcome_a, outcome_b = compare_calls(make_generator, make_generator, 4)
        self.assertEqual(outcome_a, ("return", [0, 1, 2, 3]))
        self.assertEqual(outcome_a, outcome_b)

    def test_deepcopy_isolation(self):
        mutable = [1, 2]

        def mutates(xs):
            xs.append(99)
            return sum(xs)

        compare_calls(mutates, mutates, mutable)
        self.assertEqual(mutable, [1, 2])

    def test_uncopyable_argument(self):
        lock = threading.Lock()
        with self.assertRaises(UncopyableArgumentError):
            compare_calls(add, add, lock, 1)


class AssertSameOutcomeTest(unittest.TestCase):

    def test_float_tolerance(self):
        outcome_a = ("return", 0.1 + 0.2)
        outcome_b = ("return", 0.3)
        assert_same_outcome(self, outcome_a, outcome_b)

    def test_nested_float_container(self):
        outcome_a = ("return", {"xs": [1.0, (2.0, 3.0)]})
        outcome_b = ("return", {"xs": [1.0 + 1e-10, (2.0, 3.0)]})
        assert_same_outcome(self, outcome_a, outcome_b)

    def test_float_mismatch(self):
        outcome_a = ("return", 1.0)
        outcome_b = ("return", 1.1)
        with self.assertRaises(AssertionError):
            assert_same_outcome(self, outcome_a, outcome_b)

    def test_exception_message_mismatch(self):
        outcome_a = ("raise", "builtins.ValueError", "a")
        outcome_b = ("raise", "builtins.ValueError", "b")
        with self.assertRaises(AssertionError):
            assert_same_outcome(self, outcome_a, outcome_b)

    def test_return_vs_raise(self):
        with self.assertRaises(AssertionError):
            assert_same_outcome(self, ("return", 1), ("raise", "E", "m"))


if __name__ == "__main__":
    unittest.main()
