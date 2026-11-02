"""测试批次与测试历史模型：记录一个目标函数的生成-执行轨迹。"""

from typing import Any, Optional
from pydantic import BaseModel

class TestBatch(BaseModel):
    tests: dict[str, str]
    operation: str = "generate"
    gen_model: Optional[str] = None
    gen_date: Optional[str] = None
    exec_stats: Optional[dict[str, Any]] = None
    chat_messages: Optional[list[dict[str, str]]] = None

    def add(self, test_id: str, test: str):
        self.tests[test_id] = test

    def update_stats(self, stats: dict[str, Any]):
        self.exec_stats = stats

    def update_chat_messages(self, messages: list[dict[str, str]]):
        self.chat_messages = messages

class TestRunHistory(BaseModel):
    history: list[TestBatch] = []

    def add(self, tests: TestBatch):
        self.history.append(tests)

    def update_exec_stats(self, stats: dict[str, Any]):
        self.history[-1].update_stats(stats)

    @property
    def latest_operation(self) -> str:
        return self.history[-1].operation

    @property
    def latest_tests(self) -> dict[str, str]:
        return self.history[-1].tests

    @property
    def latest_exec_stats(self) -> Optional[dict[str, Any]]:
        return self.history[-1].exec_stats

    @property
    def latest_coverage(self) -> dict[str, Any]:
        if not self.is_passing:
            return {}

        last_tests = self.history[-1]
        if last_tests.exec_stats is None:
            return {}
        if "coverage_logs" not in last_tests.exec_stats:
            return {}

        return last_tests.exec_stats["coverage_logs"][-1]

    @property
    def latest_errors(self) -> str:
        last_tests = self.history[-1]

        
        if last_tests.exec_stats is None:
            return ""

        
        if "error" in last_tests.exec_stats:
            return last_tests.exec_stats["error"]

        if "run_tests_errors" not in last_tests.exec_stats:
            return ""

        format_err = lambda e: f"{e['type']}: {e['test']}:\n{e['message']}"
        last_errors = [
            format_err(e)
            for errors in last_tests.exec_stats["run_tests_errors"].values()
            for e in errors
        ]

        return "\n".join(last_errors)

    @property
    def latest_chat_messages(self) -> list[dict[str, str]]:
        return self.history[-1].chat_messages

    @property
    def is_passing(self) -> bool:
        if len(self.history) == 0:
            return False
        last_tests = self.history[-1]
        if last_tests.exec_stats is None:
            return False
        if "run_tests_logs" not in last_tests.exec_stats:
            return False
        last_tests_logs = last_tests.exec_stats["run_tests_logs"]
        last_tests_valid = all(logs["valid"] for logs in last_tests_logs.values())

        return last_tests_valid
