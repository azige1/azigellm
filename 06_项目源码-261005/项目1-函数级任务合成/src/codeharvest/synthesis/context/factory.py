"""上下文构建的便捷入口。"""

from codeharvest.synthesis.context.dispatcher import ContextDispatcher
from codeharvest.core.prompt_context import PromptContext

def build_prompt_context(args) -> PromptContext:
    context_type, func_meth, max_context_size = args
    return ContextDispatcher.get_context(context_type, func_meth, max_context_size)
