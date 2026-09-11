"""测试合成任务对象。"""

from typing import Optional
from dataclasses import dataclass, field

from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.synthesis.tests.prompts import (
    SYSTEM_MESSAGE,
    TASK_MESSAGE_FUNCTION,
    TASK_MESSAGE_METHOD,
    OVERSAMPLE_MESSAGE,
    FIX_ERROR_MESSAGE,
    IMPROVE_COVERAGE_MESSAGE,
)

@dataclass
class TestTask:
    func_meth: FunctionRecord | MethodRecord
    generated_test: Optional[str] = None
    chat_messages: list[dict[str, str]] = field(init=False)

    def __post_init__(self):
        func_context = self.func_meth.context
        assert func_context is not None, "FunctionRecord context must not be None"

        if isinstance(self.func_meth, FunctionRecord):
            task_message = TASK_MESSAGE_FUNCTION.format(
                function_name=self.func_meth.name
            )
        elif isinstance(self.func_meth, MethodRecord):
            task_message = TASK_MESSAGE_METHOD.format(
                class_name=self.func_meth.class_name,
                method_name=self.func_meth.method_name,
            )
        else:
            raise ValueError("FunctionRecord or MethodRecord object expected")

        self.chat_messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": func_context.context,
            },
            {
                "role": "user",
                "content": task_message,
            },
        ]

    def update(
        self, generated_test: str, feedback: str = "", update_type: str = "oversample"
    ):
        self.generated_test = generated_test

        
        if self.chat_messages[-1]["role"] == "assistant":
            self.chat_messages.pop()

        if update_type == "oversample":
            self.chat_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"```python\n{generated_test}\n```",
                    },
                    {
                        "role": "user",
                        "content": OVERSAMPLE_MESSAGE,
                    },
                ]
            )

        elif update_type == "fix_error":
            self.chat_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"```python\n{generated_test}\n```",
                    },
                    {
                        "role": "user",
                        "content": FIX_ERROR_MESSAGE.format(feedback=feedback),
                    },
                ]
            )

        elif update_type == "improve_coverage":
            self.chat_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"```python\n{generated_test}\n```",
                    },
                    {
                        "role": "user",
                        "content": IMPROVE_COVERAGE_MESSAGE.format(feedback=feedback),
                    },
                ]
            )
