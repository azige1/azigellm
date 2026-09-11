"""规格合成任务对象。"""

from dataclasses import dataclass, field

from codeharvest.core.targets import FunctionTestTarget, MethodTestTarget
from codeharvest.synthesis.specs.prompts import (
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_TESTS,
    TASK_MESSAGE,
)
from codeharvest.synthesis.specs.parsing import collect_captured_types, format_io_examples

@dataclass
class SpecTask:
    func_meth: FunctionTestTarget | MethodTestTarget
    generated_test: str
    chat_messages: list[dict[str, str]] = field(init=False)

    def __post_init__(self):
        code_snipppet = self.func_meth.context
        assert self.func_meth.exec_stats is not None, f"Exec stats missing"

        captured_args = self.func_meth.exec_stats.get("captured_arg_logs", [])
        arg_types, output_type = collect_captured_types(captured_args)
        example_substring = format_io_examples(captured_args)

        self.chat_messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE_TESTS,
            },
            {
                "role": "user",
                "content": TASK_MESSAGE.format(
                    code_snipppet=code_snipppet,
                    test_code=self.generated_test,
                    argument_types=arg_types,
                    output_type=output_type,
                    example_substring=example_substring,
                    function_name=self.func_meth.name,
                ),
            },
        ]
