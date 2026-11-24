"""多轮反馈对齐测试：执行阶段丢失函数时按 id 对齐并记 fix_error，不报错。"""

import unittest
from types import SimpleNamespace
from unittest import mock

from codeharvest.synthesis.tests.loop import (
    EXECUTION_LOST_FEEDBACK,
    GenerateExecuteAgent,
)


def make_fut(func_id, passing, exec_stats=None, coverage=None, tests=None):
    return SimpleNamespace(
        id=func_id,
        is_passing=passing,
        exec_stats=exec_stats if exec_stats is not None else {},
        coverage=coverage if coverage is not None else {},
        tests=tests or {},
    )


def make_task(func_id):
    func_meth = SimpleNamespace(id=func_id, name=func_id.split(".")[-1])
    task = mock.Mock()
    task.func_meth = func_meth
    return task


class LoopAlignmentTest(unittest.TestCase):

    def _args(self):
        return SimpleNamespace(min_cov=0.8)

    def test_filter_marks_lost_function_as_fix_error(self):
        worklist = [make_task("repo.f1"), make_task("repo.f2")]
        futs_by_id = {"repo.f1": make_fut("repo.f1", passing=True, coverage={"branch_coverage_percentage": 90})}

        status_map = GenerateExecuteAgent.filter(futs_by_id, worklist, self._args())

        self.assertTrue(status_map["repo.f1"][0])
        passing, update_type, feedback = status_map["repo.f2"]
        self.assertFalse(passing)
        self.assertEqual(update_type, "fix_error")
        self.assertIn(EXECUTION_LOST_FEEDBACK, feedback)

    def test_update_worklist_keeps_only_failed_tasks(self):
        worklist = [make_task("repo.f1"), make_task("repo.f2")]
        futs_by_id = {
            "repo.f2": make_fut(
                "repo.f2",
                passing=False,
                exec_stats={"run_tests_errors": {}},
                tests={"test_0": "class T: pass"},
            )
        }
        status_map = {
            "repo.f1": (True, None, None),
            "repo.f2": (False, "fix_error", "boom"),
        }

        updated = GenerateExecuteAgent._update_worklist(worklist, futs_by_id, status_map, 1)

        self.assertEqual([t.func_meth.id for t in updated], ["repo.f2"])
        updated[0].update.assert_called_once_with("class T: pass", "boom", "fix_error")

    def test_sort_results_aligns_by_id(self):
        functions = [SimpleNamespace(id="repo.a"), SimpleNamespace(id="repo.b")]
        current_results = {
            "repo.b": make_fut("repo.b", passing=True),
            "repo.a": make_fut("repo.a", passing=True),
        }

        sorted_results = GenerateExecuteAgent._sort_results(functions, current_results)

        self.assertEqual([f.id for f in sorted_results], ["repo.a", "repo.b"])

    def test_sort_results_tolerates_missing(self):
        functions = [SimpleNamespace(id="repo.a"), SimpleNamespace(id="repo.lost")]
        current_results = {"repo.a": make_fut("repo.a", passing=True)}

        sorted_results = GenerateExecuteAgent._sort_results(functions, current_results)

        self.assertEqual([f.id for f in sorted_results], ["repo.a"])


if __name__ == "__main__":
    unittest.main()
