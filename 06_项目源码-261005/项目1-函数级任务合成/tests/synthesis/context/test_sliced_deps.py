"""测试与辅助代码。"""

import unittest

from codeharvest.synthesis.context import SlicedContextBuilder
from codeharvest.utils.io import read_callable_records
from codeharvest.workspace import INTERESTING_FUNCS_DIR

ALL_FUNCTIONS_PATH = INTERESTING_FUNCS_DIR / "all_interesting.json"
ALL_FUNCTIONS = (
    read_callable_records(ALL_FUNCTIONS_PATH) if ALL_FUNCTIONS_PATH.exists() else []
)

@unittest.skipIf(not ALL_FUNCTIONS_PATH.exists(), "ALL_FUNCTIONS_PATH not found")
class TestSlicedContextCreator(unittest.TestCase):

    def test_sliced_context(self):
        function = [
            f for f in ALL_FUNCTIONS if f.id == "klongpy.monads.eval_monad_range"
        ][0]
        function.repo.repo_id = function.repo.local_repo_path.split("/")[-1]
        context_creator = SlicedContextBuilder(function, 6000)
        sliced_context = context_creator.get_context()

        self.assertEqual(sliced_context.context.count("```python"), 3)
        self.assertNotIn("create_monad_functions", sliced_context.context)
        self.assertIn("str_to_chr_arr", sliced_context.context)

if __name__ == "__main__":
    unittest.main()
