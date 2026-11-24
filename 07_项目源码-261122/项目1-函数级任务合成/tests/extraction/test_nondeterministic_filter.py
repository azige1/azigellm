"""nondeterministic_filter 的单元测试。"""

import ast
import unittest

from codeharvest.extraction.targets.nondeterministic_filter import NondeterministicFilter


class NondeterministicFilterTest(unittest.TestCase):

    def _filtered_names(self, *sources: str) -> set:
        module = ast.parse("\n\n".join(sources))
        functions = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        kept = NondeterministicFilter.filter_functions(functions, module)
        return {node.name for node in kept}

    def test_pure_kept(self):
        self.assertIn("pure", self._filtered_names("def pure(x):\n    return x + 1\n"))

    def test_time_time_removed(self):
        self.assertNotIn(
            "stamp", self._filtered_names("def stamp():\n    import time\n    return time.time()\n")
        )

    def test_datetime_now_removed(self):
        self.assertNotIn(
            "now",
            self._filtered_names(
                "def now():\n    import datetime\n    return datetime.datetime.now()\n"
            ),
        )

    def test_uuid_removed(self):
        self.assertNotIn(
            "gen",
            self._filtered_names("def gen():\n    import uuid\n    return uuid.uuid4()\n"),
        )

    def test_seeded_random_kept(self):
        self.assertIn(
            "pick",
            self._filtered_names(
                "def pick(xs):\n"
                "    import random\n"
                "    random.seed(42)\n"
                "    return random.choice(xs)\n"
            ),
        )

    def test_unseeded_random_removed(self):
        self.assertNotIn(
            "pick",
            self._filtered_names(
                "def pick(xs):\n    import random\n    return random.choice(xs)\n"
            ),
        )

    def test_network_removed(self):
        self.assertNotIn(
            "fetch",
            self._filtered_names(
                "def fetch(url):\n    import requests\n    return requests.get(url)\n"
            ),
        )

    def test_nested_callee_removed(self):
        sources = (
            "def _id():\n    import uuid\n    return uuid.uuid4()\n",
            "def make_key(x):\n    return f'{x}-{_id()}'\n",
        )
        self.assertNotIn("make_key", self._filtered_names(*sources))

    def test_os_urandom_removed(self):
        self.assertNotIn(
            "rand", self._filtered_names("def rand():\n    import os\n    return os.urandom(4)\n")
        )


if __name__ == "__main__":
    unittest.main()
