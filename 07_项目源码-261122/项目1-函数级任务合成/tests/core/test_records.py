"""测试与辅助代码。"""

import unittest
import json
import os
from unittest.mock import patch, mock_open
from codeharvest.core.repository import RepoRecord
from codeharvest.core.code_module import ModuleRecord, ModuleKind
from codeharvest.core.identifiers import SymbolId
from codeharvest.core.functions import FunctionRecord
from codeharvest.core.source_file import FileRecord
from codeharvest.core.callgraph import CallGraphModel, CodeElementKind
from codeharvest.workspace import REPOS_DIR

repo_json = json.dumps(
    {
        "repo_org": "test_org",
        "repo_name": "test_repo",
        "repo_id": "123456",
        "local_repo_path": "repos/test_repo",
    }
)

module_json = json.dumps(
    {
        "module_id": {"identifier": "test.module"},
        "module_type": "file",
        "repo": json.loads(repo_json),
    }
)

function_json = json.dumps(
    {
        "function_id": {"identifier": "test.module.function"},
        "file": {"file_module": json.loads(module_json)},
        "function_name": "function",
        "function_code": "def function(): pass",
    }
)

callgraph_json = json.dumps(
    {
        "graph": {"test.module.function": ["test.mod2.callee1", "test.mod3.callee2"]},
        "id2type": {
            "test.module.function": "FUNCTION",
            "test.mod2.callee1": "FUNCTION",
            "test.mod3.callee2": "FUNCTION",
        },
    }
)

class TestRepo(unittest.TestCase):
    def setUp(self):
        self.repo_data = json.loads(repo_json)
        self.repo = RepoRecord(**self.repo_data)

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=callgraph_json,
    )
    def test_callgraph(self, mock_open, mock_exists):
        callgraph = self.repo.callgraph
        function_id = SymbolId(identifier="test.module.function")
        idtype = callgraph.get_type(function_id)
        self.assertIsInstance(callgraph, CallGraphModel)
        self.assertIn(function_id, callgraph)
        self.assertEqual(idtype, CodeElementKind["FUNCTION"])

class TestModule(unittest.TestCase):
    def setUp(self):
        self.module_data = json.loads(module_json)
        self.module = ModuleRecord(**self.module_data)

    @patch("os.path.exists", return_value=True)
    def test_local_path(self, mock_exists):
        expected_path = f"{REPOS_DIR}/123456/test/module"
        self.assertEqual(self.module.local_path, expected_path)

    @patch("os.path.exists", return_value=True)
    def test_exists(self, mock_exists):
        self.assertTrue(self.module.exists())

class TestFunction(unittest.TestCase):
    def setUp(self):
        self.function_data = json.loads(function_json)
        self.function = FunctionRecord(**self.function_data)

    @patch("os.path.exists", return_value=True)
    def test_file_path(self, mock_exists):
        expected_path = f"{REPOS_DIR}/123456/test/module"
        self.assertEqual(self.function.file_path, expected_path)

    def test_repo_id(self):
        self.assertEqual(self.function.repo.repo_id, "123456")

    def test_module(self):
        self.assertIsInstance(self.function.module, ModuleRecord)
        self.assertEqual(self.function.module.module_id.identifier, "test.module")

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=callgraph_json,
    )
    def test_callees(self, mock_open, mock_exists):
        self.assertIsInstance(self.function.callees, list)
        self.assertEqual(len(self.function.callees), 2)
        self.assertIsInstance(self.function.callees[0], FunctionRecord)
        self.assertEqual(self.function.callees[0].id, "test.mod2.callee1")

        assert isinstance(self.function.callees[1], FunctionRecord)
        self.assertEqual(self.function.callees[1].function_name, "callee2")

if __name__ == "__main__":
    unittest.main()
