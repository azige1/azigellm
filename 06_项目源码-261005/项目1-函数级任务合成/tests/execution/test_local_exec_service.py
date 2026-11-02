"""本地执行服务测试：临时小仓库跑 init/submit、自等价拦截与 codegen 模式替换。"""

import json
import os
import tempfile
import textwrap
import unittest

from codeharvest.execution.local_exec_service import RepoExecService


GOOD_TEST = textwrap.dedent(
    """
    import unittest
    from fut_module import add, reference_add
    from equivalence_helpers import compare_calls, assert_same_outcome

    class TestAdd(unittest.TestCase):
        def test_positive(self):
            a, b = compare_calls(add, reference_add, 1, 2)
            assert_same_outcome(self, a, b)

        def test_negative(self):
            a, b = compare_calls(add, reference_add, -1, 1)
            assert_same_outcome(self, a, b)
    """
).strip()

BAD_TEST = textwrap.dedent(
    """
    import unittest
    from fut_module import add, reference_add

    class TestAddBroken(unittest.TestCase):
        def test_wrong_expectation(self):
            self.assertEqual(add(1, 2), 4)
    """
).strip()


class RepoExecServiceTest(unittest.TestCase):

    def setUp(self):
        self.repo_dir = tempfile.mkdtemp(prefix="codeharvest_test_repo_")
        with open(os.path.join(self.repo_dir, "mod.py"), "w") as f:
            f.write("def add(a, b):\n    return a + b\n")

    def _make_service(self, tests):
        service = RepoExecService()
        response = service.setup_repo(
            json.dumps({"repo_id": None, "repo_path": self.repo_dir})
        )
        self.assertEqual(response["error"], "")
        response = service.setup_function(
            json.dumps({"funclass_names": ["add"], "file_path": "mod.py"})
        )
        self.assertEqual(response["error"], "")
        response = service.setup_test(json.dumps({"generated_tests": tests}))
        self.assertEqual(response["error"], "")
        return service

    def _submit_logs(self, service):
        response = service.submit()
        self.assertIn("logs", response)
        return json.loads(response["logs"])

    def test_init_and_submit_self_equivalence(self):
        service = self._make_service({"test_0": GOOD_TEST})
        init_response = service.init()
        self.assertEqual(init_response["error"], "")

        logs = self._submit_logs(service)
        self.assertEqual(logs["verdict"], "passed")
        self.assertTrue(logs["run_tests_logs"]["test_0"]["valid"])
        self.assertEqual(logs["run_tests_logs"]["test_0"]["passed_count"], 2)
        self.assertEqual(logs["run_tests_logs"]["test_0"]["failed_count"], 0)

        coverage = logs["coverage_logs"][0]
        self.assertEqual(coverage["start_line"], 1)
        self.assertEqual(coverage["end_line"], 2)
        self.assertGreaterEqual(coverage["branch_coverage_percentage"], 0)
        self.assertLessEqual(coverage["branch_coverage_percentage"], 100)

        captured = logs["captured_arg_logs"]
        self.assertTrue(any("add" in key for key in captured))
        service.stop_server()

    def test_self_equivalence_intercepts_bad_test(self):
        service = self._make_service({"test_0": BAD_TEST})
        init_response = service.init()
        self.assertEqual(init_response["error"], "")

        logs = self._submit_logs(service)
        self.assertEqual(logs["verdict"], "test_defect")
        self.assertFalse(logs["run_tests_logs"]["test_0"]["valid"])
        self.assertEqual(len(logs["run_tests_errors"]["test_0"]), 1)
        error = logs["run_tests_errors"]["test_0"][0]
        self.assertIn("test_wrong_expectation", error["test"])
        self.assertIsNotNone(error["line"])
        service.stop_server()

    def test_codegen_mode_replaces_candidate(self):
        service = self._make_service({"test_0": GOOD_TEST})
        init_response = service.init()
        self.assertEqual(init_response["error"], "")

        service.setup_codegen_mode()
        wrong_impl = "def add(a, b):\n    return a - b\n"
        exec_response = service.execute(wrong_impl)
        self.assertEqual(exec_response["error"], "")

        logs = self._submit_logs(service)
        self.assertEqual(logs["verdict"], "test_defect")
        self.assertFalse(logs["run_tests_logs"]["test_0"]["valid"])
        service.stop_server()

    def test_codegen_method_replacement_keeps_other_members(self):
        with open(os.path.join(self.repo_dir, "shapes.py"), "w") as f:
            f.write(
                "class Box:\n"
                "    def __init__(self, w):\n"
                "        self.w = w\n"
                "    def area(self):\n"
                "        return self.w * self.w\n"
            )
        method_test = textwrap.dedent(
            """
            import unittest
            from fut_module import Box, reference_Box
            from equivalence_helpers import compare_calls, assert_same_outcome

            class TestArea(unittest.TestCase):
                def test_area(self):
                    a, b = compare_calls(Box(3).area, reference_Box(3).area)
                    assert_same_outcome(self, a, b)
            """
        ).strip()
        service = RepoExecService()
        service.setup_repo(json.dumps({"repo_id": None, "repo_path": self.repo_dir}))
        service.setup_function(
            json.dumps({"funclass_names": ["Box.area"], "file_path": "shapes.py"})
        )
        service.setup_test(json.dumps({"generated_tests": {"test_0": method_test}}))
        self.assertEqual(service.init()["error"], "")
        service.setup_codegen_mode()
        # 候选只给出方法本身，__init__ 必须保留
        response = service.execute("def area(self):\n    return self.w ** 2\n")
        self.assertEqual(response["error"], "")
        logs = self._submit_logs(service)
        self.assertEqual(logs["verdict"], "passed")
        service.stop_server()

    def test_relative_import_rewritten_in_package(self):
        package_dir = os.path.join(self.repo_dir, "pkg")
        os.makedirs(package_dir)
        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(package_dir, "helpers.py"), "w") as f:
            f.write("def double(x):\n    return 2 * x\n")
        with open(os.path.join(package_dir, "core.py"), "w") as f:
            f.write("from .helpers import (\n    double,\n)\n\ndef quad(x):\n    return double(double(x))\n")
        quad_test = textwrap.dedent(
            """
            import unittest
            from fut_module import quad, reference_quad
            from equivalence_helpers import compare_calls, assert_same_outcome

            class TestQuad(unittest.TestCase):
                def test_quad(self):
                    a, b = compare_calls(quad, reference_quad, 3)
                    assert_same_outcome(self, a, b)
            """
        ).strip()
        service = RepoExecService()
        service.setup_repo(json.dumps({"repo_id": None, "repo_path": self.repo_dir}))
        service.setup_function(
            json.dumps({"funclass_names": ["quad"], "file_path": "pkg/core.py"})
        )
        service.setup_test(json.dumps({"generated_tests": {"test_0": quad_test}}))
        self.assertEqual(service.init()["error"], "")
        logs = self._submit_logs(service)
        self.assertEqual(logs["verdict"], "passed")
        # 行号保持：quad 在原文件第 5 行
        self.assertEqual(logs["coverage_logs"][0]["start_line"], 5)
        service.stop_server()

    def test_execute_rejects_missing_definition(self):
        service = self._make_service({"test_0": GOOD_TEST})
        service.init()
        service.setup_codegen_mode()
        response = service.execute("def unrelated():\n    pass\n")
        self.assertNotEqual(response["error"], "")
        service.stop_server()


if __name__ == "__main__":
    unittest.main()
