"""任务导出测试：按仓库切分无交叉、黑名单过滤。"""

import os
import tempfile
import unittest

from codeharvest.synthesis.tasks.export import (
    assign_split,
    is_blocked,
    load_blocklist,
)


class SplitTest(unittest.TestCase):

    def test_split_is_deterministic_and_disjoint(self):
        ratios = [0.1, 0.5]
        repo_ids = [f"org{i}___repo{i}" for i in range(50)]
        for ratio in ratios:
            splits = [assign_split(repo_id, ratio) for repo_id in repo_ids]
            self.assertTrue(all(s in ("train", "eval") for s in splits))
            self.assertEqual(splits, [assign_split(r, ratio) for r in repo_ids])
            eval_count = sum(1 for s in splits if s == "eval")
            self.assertGreater(eval_count, 0)
            self.assertLess(eval_count, len(repo_ids))

    def test_same_repo_same_split(self):
        self.assertEqual(
            assign_split("org___django", 0.1), assign_split("org___django", 0.1)
        )


class BlocklistTest(unittest.TestCase):

    def _write_blocklist(self, lines: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(lines)
        self.addCleanup(os.unlink, path)
        return path

    def test_blocklist_filtering(self):
        path = self._write_blocklist(
            "# 注释行\nDjango/Django\nscikit-learn/scikit-learn\n")
        blocklist = load_blocklist(path)
        self.assertTrue(is_blocked("someorg___django", blocklist))
        self.assertTrue(is_blocked("scikit-learn___scikit-learn", blocklist))
        self.assertFalse(is_blocked("sdispater___tomlkit", blocklist))

    def test_blocklist_ignores_org_and_case(self):
        path = self._write_blocklist("PyCQA/attrs\n")
        blocklist = load_blocklist(path)
        self.assertTrue(is_blocked("whatever___Attrs", blocklist))


if __name__ == "__main__":
    unittest.main()
