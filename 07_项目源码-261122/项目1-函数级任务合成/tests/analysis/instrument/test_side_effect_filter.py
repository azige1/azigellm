"""side_effect_filter 的单元测试。"""

import ast
import unittest

from codeharvest.analysis.instrument.side_effect_filter import SideEffectFilter


def parse_function(source: str):
    tree = ast.parse(source)
    return tree.body[0], tree


class SideEffectFilterTest(unittest.TestCase):

    def _has_side_effects(self, source: str) -> bool:
        node, tree = parse_function(source)
        module_functions = SideEffectFilter.module_function_table(tree)
        return SideEffectFilter.has_side_effects(node, module_functions, tree)

    def test_pure_function_clean(self):
        self.assertFalse(
            self._has_side_effects("def f(x):\n    return x * 2\n")
        )

    def test_write_open_flagged(self):
        self.assertTrue(
            self._has_side_effects(
                "def f(path):\n    with open(path, 'w') as fh:\n        fh.write('x')\n"
            )
        )

    def test_read_open_clean(self):
        self.assertFalse(
            self._has_side_effects("def f(path):\n    return open(path).read()\n")
        )

    def test_subprocess_flagged(self):
        self.assertTrue(
            self._has_side_effects(
                "def f(cmd):\n    import subprocess\n    return subprocess.run(cmd)\n"
            )
        )

    def test_global_statement_flagged(self):
        self.assertTrue(
            self._has_side_effects("def f(x):\n    global counter\n    counter += x\n")
        )

    def test_name_prefix_flagged(self):
        self.assertTrue(
            self._has_side_effects("def save_result(x):\n    return x\n")
        )

    def test_callee_in_module_flagged(self):
        source = (
            "def _persist(x):\n"
            "    with open('out.txt', 'w') as fh:\n"
            "        fh.write(x)\n"
            "def handler(x):\n"
            "    return _persist(x)\n"
        )
        tree = ast.parse(source)
        handler = tree.body[1]
        module_functions = SideEffectFilter.module_function_table(tree)
        self.assertTrue(SideEffectFilter.has_side_effects(handler, module_functions, tree))

    def test_local_assignment_clean(self):
        self.assertFalse(
            self._has_side_effects(
                "def f(items):\n    total = 0\n    for x in items:\n        total += x\n    return total\n"
            )
        )

    def test_module_level_cache_write_flagged(self):
        source = "CACHE = {}\ndef f(k, v):\n    CACHE[k] = v\n    return v\n"
        tree = ast.parse(source)
        function_node = tree.body[1]
        module_functions = SideEffectFilter.module_function_table(tree)
        self.assertTrue(SideEffectFilter.has_side_effects(function_node, module_functions, tree))

    def test_self_attribute_write_clean(self):
        source = "class A:\n    def set(self, v):\n        self.v = v\n        return v\n"
        tree = ast.parse(source)
        method = tree.body[0].body[0]
        self.assertFalse(SideEffectFilter.has_side_effects(method, {}, tree))


if __name__ == "__main__":
    unittest.main()
