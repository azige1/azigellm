"""测试与辅助代码。"""

import json
import os
import unittest
from unittest.mock import patch, mock_open
import tempfile

from codeharvest.core.identifiers import SymbolId
from codeharvest.core.code_module import ModuleRecord
from codeharvest.core.repository import RepoRecord
from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.core.classes import ClassRecord

from codeharvest.core.source_file import FileRecord
from codeharvest.core.callgraph import CallGraphModel

from codeharvest.analysis.callgraph.query import CallGraphQuery

callgraph_json = json.dumps(
    {
        "graph": {
            "src.utils.foo": ["src.utils.bar"],
            "src.classes.MyClass": ["<builtin>.print"],
            "src.classes.MyClass.my_method": ["src.utils.baz"],
            "src.classes.MyClass.my_method2": ["src.classes.MyClass.my_method"],
        },
        "id2type": {
            "src.utils.foo": "FUNCTION",
            "src.utils.bar": "FUNCTION",
            "src.utils.baz": "FUNCTION",
            "src.classes.MyClass": "CLASS",
            "src.classes.MyClass.my_method": "METHOD",
            "src.classes.MyClass.my_method2": "METHOD",
            "<builtin>.print": "BUILTIN",
        },
    }
)

class TestCallGraphExplorer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.repo_path = self.test_dir.name

        self.repo = RepoRecord(
            repo_org="",
            repo_name="test_repo",
            repo_id="",
            local_repo_path=self.repo_path,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=callgraph_json,
    )
    def test_get_function_callees(self, mock_open, mock_exists):
        cg_explorer = CallGraphQuery(self.repo)
        callees = cg_explorer.callees_for_identifier("src.utils.foo")
        self.assertEqual(len(callees), 1)
        self.assertIsInstance(callees[0], FunctionRecord)

        if isinstance(callees[0], FunctionRecord):
            self.assertEqual(callees[0].function_id.identifier, "src.utils.bar")

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=callgraph_json,
    )
    def test_get_class_method_callees(self, mock_open, mock_exists):
        cg_explorer = CallGraphQuery(self.repo)
        callees = cg_explorer.callees_for_identifier(
            "src.classes.MyClass.my_method2"
        )
        self.assertEqual(len(callees), 1)
        self.assertIsInstance(callees[0], MethodRecord)

        if isinstance(callees[0], MethodRecord):
            self.assertEqual(
                callees[0].method_id.identifier, "src.classes.MyClass.my_method"
            )

        
        caller_id = SymbolId(identifier="src.classes.MyClass.my_method")
        caller = MethodRecord.from_id_and_repo(caller_id, self.repo)

        if isinstance(callees[0], MethodRecord):
            self.assertEqual(callees[0].parent_class_id, caller.parent_class_id)

        
        class_id = SymbolId(identifier="src.classes.MyClass")
        class_ = ClassRecord.from_id_and_repo(class_id, self.repo)
        class_methods_ids = class_.method_ids
        self.assertIn(caller.method_id, class_methods_ids)

        if isinstance(callees[0], MethodRecord):
            self.assertIn(callees[0].method_id, class_methods_ids)

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=callgraph_json,
    )
    def test_get_class_callees(self, mock_open, mock_exists):
        cg_explorer = CallGraphQuery(self.repo)
        callees = cg_explorer.callees_for_identifier("src.classes.MyClass")
        self.assertEqual(len(callees), 3)
        expected_callees = [
            "<builtin>.print",
            "src.utils.baz",
            "src.classes.MyClass.my_method",
        ]
        for callee in callees:
            self.assertIsInstance(callee, (MethodRecord, FunctionRecord))
            if isinstance(callee, MethodRecord):
                self.assertIn(callee.method_id.identifier, expected_callees)
            elif isinstance(callee, FunctionRecord):
                self.assertIn(callee.function_id.identifier, expected_callees)

if __name__ == "__main__":
    unittest.main()
