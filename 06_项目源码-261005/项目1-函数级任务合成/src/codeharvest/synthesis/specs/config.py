"""规格合成的参数配置。"""

from codeharvest.llm.config import LLMConfig
from pydantic import Field

class SpecGenConfig(LLMConfig):
    temperature: float = Field(
        0.3,
        description="The temperature for the spec refinement request",
    )

    in_file: str = Field(
        None,
        description="The input file for the spec generator",
    )

    exp_id: str = Field(
        "temp",
        description="Experiment ID used for prefixing the generated file.",
    )

    limit: int | None = Field(
        None,
        description="Optional cap on the number of functions processed",
    )
