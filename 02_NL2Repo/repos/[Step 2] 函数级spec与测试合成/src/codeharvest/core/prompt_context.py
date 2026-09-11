"""提示上下文模型：发给大模型的代码上下文载体。"""

from enum import Enum

from pydantic import BaseModel

class PromptContext(BaseModel):
    context_type: str
    context: str
