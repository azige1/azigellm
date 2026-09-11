"""模型注册表：模型名与后端类型。"""

from enum import Enum
from dataclasses import dataclass

class ModelBackend(Enum):

    OpenAI = "openai"
    Gemini = "gemini"
    Claude3 = "claude3"
    VLLM = "vllm"

@dataclass
class ModelSpec:
    model_name: str
    style: ModelBackend
    context_length: int = -1

MODEL_REGISTRY: list[ModelSpec] = [
    
    
    
    
    ModelSpec(
        model_name="gpt-3.5-turbo-0613",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-3.5-turbo-1106",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-3.5-turbo-0125",
        style=ModelBackend.OpenAI,
    ),
    
    
    
    
    ModelSpec(
        model_name="gpt-3.5-turbo-16k-0125",
        style=ModelBackend.OpenAI,
    ),
    
    
    
    
    ModelSpec(
        model_name="gpt-4-0613",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-4-0314",
        style=ModelBackend.OpenAI,
    ),
    
    
    
    
    ModelSpec(
        model_name="gpt-4-32k-0613",
        style=ModelBackend.OpenAI,
    ),
    
    
    
    
    ModelSpec(
        model_name="gpt-4-1106-preview",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-4-0125-preview",
        style=ModelBackend.OpenAI,
    ),
    
    
    
    
    ModelSpec(
        model_name="gpt-4-turbo-2024-04-09",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-4o-mini",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="gpt-4o",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="o1-preview",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="o1-mini",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="o3-mini",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="o3",
        style=ModelBackend.OpenAI,
    ),
    ModelSpec(
        model_name="o4-mini",
        style=ModelBackend.OpenAI,
    ),
]
