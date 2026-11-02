"""测试目标模型：被测函数 / 被测方法及其测试历史。"""

from typing import Any, Optional

from pydantic import BaseModel

from codeharvest.core import FunctionRecord, MethodRecord
from codeharvest.core.test_history import TestRunHistory, TestBatch

class TestTargetBase(BaseModel):
    test_history: TestRunHistory

    @property
    def tests(self) -> dict[str, str]:
        return self.test_history.latest_tests

    @property
    def operation(self) -> str:
        return self.test_history.latest_operation

    @property
    def exec_stats(self) -> Optional[dict[str, Any]]:
        return self.test_history.latest_exec_stats

    @property
    def coverage(self) -> dict[str, Any]:
        return self.test_history.latest_coverage

    @property
    def errors(self) -> str:
        return self.test_history.latest_errors

    def update_history(self, tests: TestBatch):
        self.test_history.add(tests)

    def update_exec_stats(self, stats: dict[str, Any]):
        self.test_history.update_exec_stats(stats)

    @property
    def is_passing(self) -> bool:
        return self.test_history.is_passing

class FunctionTestTarget(TestTargetBase, FunctionRecord):
    @classmethod
    def from_function_and_history(
        cls, function: FunctionRecord, history: Optional[TestRunHistory] = None
    ):
        if history is None:
            history = TestRunHistory()
        function_dump = function.model_dump()
        function_dump.update({"test_history": history})
        return cls(**function_dump)

    @classmethod
    def from_function(cls, function: FunctionRecord):
        return cls.from_function_and_history(function)

    @property
    def execution_fut_data(self) -> tuple[str, str]:
        return f"{self.name}", self.file.relative_file_path

class MethodTestTarget(TestTargetBase, MethodRecord):
    @classmethod
    def from_method_and_history(
        cls, method: MethodRecord, history: Optional[TestRunHistory] = None
    ):
        if history is None:
            history = TestRunHistory()
        method_dump = method.model_dump()
        method_dump.update({"test_history": history})
        return cls(**method_dump)

    @classmethod
    def from_method(cls, method: MethodRecord):
        return cls.from_method_and_history(method)

    @property
    def execution_fut_data(self) -> tuple[str, str]:
        return (
            f"{self.parent_class.class_name}.{self.name}",
            self.file.relative_file_path,
        )

def make_code_under_test(obj: FunctionRecord | MethodRecord):
    if isinstance(obj, FunctionRecord):
        return FunctionTestTarget.from_function(obj)
    elif isinstance(obj, MethodRecord):
        return MethodTestTarget.from_method(obj)
    else:
        raise TypeError("obj must be a FunctionRecord or MethodRecord instance")
