"""测试与辅助代码。"""

import unittest

from codeharvest.synthesis.context import ContextBuilder
from codeharvest.utils.io import read_callable_records
from codeharvest.workspace import INTERESTING_FUNCS_DIR

ALL_FUNCTIONS_PATH = INTERESTING_FUNCS_DIR / "all_interesting.json"
ALL_FUNCTIONS = (
    read_callable_records(ALL_FUNCTIONS_PATH) if ALL_FUNCTIONS_PATH.exists() else []
)

@unittest.skipIf(not ALL_FUNCTIONS_PATH.exists(), "ALL_FUNCTIONS_PATH not found")
class TestNaiveContextCreator(unittest.TestCase):

    def test_naive_context(self):
        function = [
            f for f in ALL_FUNCTIONS if f.id == "klongpy.monads.eval_monad_range"
        ][0]

        context_creator = ContextBuilder(function, 6000)
        context_creator.construct_context()
        naive_context = context_creator.get_context()

        self.assertEqual(naive_context.context.count("```python"), 1)
        self.assertIn("eval_monad_range", naive_context.context)
        self.assertEqual(naive_context.context.count("def "), 1)

if __name__ == "__main__":
    unittest.main()
