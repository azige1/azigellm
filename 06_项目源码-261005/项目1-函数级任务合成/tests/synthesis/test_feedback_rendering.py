"""反馈渲染测试：fix_error 含行号与行源码，improve_coverage 含分支计数与条件表达式。"""

import os
import tempfile
import textwrap
import unittest
from types import SimpleNamespace

from codeharvest.synthesis.tests.feedback import (
    render_coverage_feedback,
    render_error_feedback,
)


def make_fut(exec_stats=None, coverage=None, tests=None, source_path=None):
    return SimpleNamespace(
        id="repo.mod.f",
        exec_stats=exec_stats or {},
        coverage=coverage or {},
        tests=tests or {},
        file_path=source_path or "",
    )


class ErrorFeedbackTest(unittest.TestCase):

    def test_feedback_contains_line_and_source(self):
        test_source = textwrap.dedent(
            """
            import unittest
            class T(unittest.TestCase):
                def test_boom(self):
                    raise RuntimeError("kaboom")
            """
        ).strip()
        fut = make_fut(
            exec_stats={
                "run_tests_errors": {
                    "test_0": [
                        {
                            "type": "error",
                            "test": "test_boom",
                            "message": "Traceback (most recent call last):\n"
                            '  File "/x/test_0.py", line 4, in test_boom\n'
                            "RuntimeError: kaboom",
                            "line": 4,
                        }
                    ]
                }
            },
            tests={"test_0": test_source},
        )
        feedback = render_error_feedback(fut)
        self.assertIn("test_boom", feedback)
        self.assertIn("RuntimeError: kaboom", feedback)
        self.assertIn("Failing line: 4", feedback)
        self.assertIn("raise RuntimeError(\"kaboom\")", feedback)

    def test_feedback_without_errors_falls_back(self):
        fut = make_fut(exec_stats={"error": "init failed"})
        self.assertIn("init failed", render_error_feedback(fut))


class CoverageFeedbackTest(unittest.TestCase):

    def _write_source(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        self.addCleanup(os.unlink, path)
        return path

    def test_feedback_contains_branch_count_and_condition(self):
        source_path = self._write_source(
            "def f(x):\n    if x > 0:\n        return 1\n    return 0\n"
        )
        fut = make_fut(
            coverage={
                "covered_branches": 1,
                "total_branches": 2,
                "start_line": 1,
                "end_line": 4,
                "unexecuted_lines": [3],
                # 执行日志经 JSON 往返后行号键为字符串
                "unevaluated_branches": {"2": ["x > 0", [3]]},
            },
            source_path=source_path,
        )
        feedback = render_coverage_feedback(fut)
        self.assertIn("Covered branches: 1 / 2", feedback)
        self.assertIn("# UNCOVERED BRANCH: x > 0", feedback)
        self.assertIn("missing jump to line(s): 3", feedback)


if __name__ == "__main__":
    unittest.main()
