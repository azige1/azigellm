"""上下文构建：为大模型准备目标函数的代码上下文。"""

from codeharvest.synthesis.context.builder import ContextBuilder
from codeharvest.synthesis.context.full_repo import FullRepoContextBuilder
from codeharvest.synthesis.context.sliced_deps import SlicedContextBuilder
from codeharvest.synthesis.context.formatting import ContextRenderer, ContextOutputFormat
from codeharvest.synthesis.context.dispatcher import ContextDispatcher
from codeharvest.synthesis.context.factory import build_prompt_context
