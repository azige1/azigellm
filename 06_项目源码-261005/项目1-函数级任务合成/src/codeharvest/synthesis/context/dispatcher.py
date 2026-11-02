"""按配置分派上下文构建器。"""

from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.core.prompt_context import PromptContext
from codeharvest.synthesis.context import (
    ContextBuilder,
    FullRepoContextBuilder,
    SlicedContextBuilder,
)

class ContextDispatcher:

    @staticmethod
    def get_context(
        context_type: str, func_meth: FunctionRecord | MethodRecord, max_context_size: int
    ) -> PromptContext:

        if context_type == "naive":
            nc = ContextBuilder(func_meth, max_context_size)
            nc.construct_context()
            return nc.get_context()

        elif context_type == "full":
            return FullRepoContextBuilder(func_meth, max_context_size).get_context()

        elif context_type == "sliced":
            return SlicedContextBuilder(func_meth, max_context_size).get_context()

        else:
            raise ValueError(f"Invalid context type: {context_type}")
