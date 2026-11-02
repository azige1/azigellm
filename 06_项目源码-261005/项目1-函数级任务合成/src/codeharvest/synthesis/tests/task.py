"""测试合成任务对象。"""

from typing import Optional
from dataclasses import dataclass, field

from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.synthesis.tests.prompts import (
    SYSTEM_MESSAGE_EQUIVALENCE,
    TASK_MESSAGE_FUNCTION_EQUIVALENCE,
    TASK_MESSAGE_METHOD_EQUIVALENCE,
    SYSTEM_MESSAGE_LITERAL,
    TASK_MESSAGE_FUNCTION_LITERAL,
    TASK_MESSAGE_METHOD_LITERAL,
    CAPTURED_CALLS_MESSAGE,
    OVERSAMPLE_MESSAGE,
    FIX_ERROR_MESSAGE,
    IMPROVE_COVERAGE_MESSAGE,
)

MAX_CAPTURED_CALLS_IN_PROMPT = 3


def _format_captured_call(record: dict) -> str:
    """把一条采集记录渲染为一行入参 / 返回（或异常）/ 调用者。"""
    inputs = ", ".join(f"{k}={v}" for k, v in record.get("inputs", {}).items())
    if "exception_type" in record:
        outcome = f"raises {record['exception_type']}: {record.get('exception_message', '')}"
    else:
        outcome = f"returns {record.get('output')} ({record.get('output_type', '')})"
    caller = record.get("caller_info") or {}
    caller_text = f"{caller.get('func_name', '?')}:{caller.get('lineno', '?')}"
    return f"- inputs: {inputs}\n  outcome: {outcome}\n  caller: {caller_text}"


@dataclass
class TestTask:
    func_meth: FunctionRecord | MethodRecord
    generated_test: Optional[str] = None
    test_style: str = "equivalence"
    captured_calls: Optional[list[dict]] = None
    chat_messages: list[dict[str, str]] = field(init=False)

    def __post_init__(self):
        func_context = self.func_meth.context
        assert func_context is not None, "FunctionRecord context must not be None"

        if self.test_style == "literal":
            system_message = SYSTEM_MESSAGE_LITERAL
            task_message_function = TASK_MESSAGE_FUNCTION_LITERAL
            task_message_method = TASK_MESSAGE_METHOD_LITERAL
        else:
            system_message = SYSTEM_MESSAGE_EQUIVALENCE
            task_message_function = TASK_MESSAGE_FUNCTION_EQUIVALENCE
            task_message_method = TASK_MESSAGE_METHOD_EQUIVALENCE

        if isinstance(self.func_meth, FunctionRecord):
            task_message = task_message_function.format(
                function_name=self.func_meth.name
            )
        elif isinstance(self.func_meth, MethodRecord):
            task_message = task_message_method.format(
                class_name=self.func_meth.class_name,
                method_name=self.func_meth.method_name,
            )
        else:
            raise ValueError("FunctionRecord or MethodRecord object expected")

        self.chat_messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": func_context.context,
            },
        ]

        # 有采集记录时，在上下文之后、任务消息之前插入真实调用观察
        if self.captured_calls:
            calls_text = "\n\n".join(
                _format_captured_call(record)
                for record in self.captured_calls[:MAX_CAPTURED_CALLS_IN_PROMPT]
            )
            self.chat_messages.append(
                {
                    "role": "user",
                    "content": CAPTURED_CALLS_MESSAGE.format(
                        function_name=self.func_meth.name, captured_calls=calls_text
                    ),
                }
            )

        self.chat_messages.append(
            {
                "role": "user",
                "content": task_message,
            }
        )

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
