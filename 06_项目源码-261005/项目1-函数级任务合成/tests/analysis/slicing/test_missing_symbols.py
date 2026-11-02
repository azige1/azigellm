"""missing_symbols 的单元测试。"""

import unittest

from codeharvest.analysis.slicing.missing_symbols import find_missing_symbols


class MissingSymbolsTest(unittest.TestCase):

    def test_no_missing(self):
        code = (
            "import math\n"
            "def area(r):\n"
            "    return math.pi * r * r\n"
        )
        self.assertEqual(find_missing_symbols(code, ["area"]), [])

    def test_missing_global(self):
        code = (
            "def f(x):\n"
            "    return helper(x) + UNDEFINED_CONST\n"
        )
        missing = find_missing_symbols(code, ["f"])
        self.assertIn("helper", missing)
        self.assertIn("UNDEFINED_CONST", missing)

    def test_builtin_not_missing(self):
        code = "def g(xs):\n    return len(xs) + max(xs)\n"
        self.assertEqual(find_missing_symbols(code, ["g"]), [])

    def test_class_method_entry(self):
        code = (
            "class A:\n"
            "    def m(self):\n"
            "        return missing_fn()\n"
        )
        self.assertEqual(find_missing_symbols(code, ["m"]), ["missing_fn"])

    def test_syntax_error_returns_empty(self):
        self.assertEqual(find_missing_symbols("def broken(:", ["broken"]), [])


if __name__ == "__main__":
    unittest.main()
