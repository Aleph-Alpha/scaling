import math
from enum import Enum
from typing import List

from pydantic import Field

from scaling.core.config import BaseConfig


class LoRAModuleType(Enum):
    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    DENSE = "dense"


class LoRaConfig(BaseConfig):
    name: str = Field(default="lora", description="")

    rank: int = Field(default=64, description="Intrinsic Rank of the Lora Adapter over all heads.")

    parallel_modules: List[LoRAModuleType] = Field(
        default=[
            LoRAModuleType.DENSE,
            LoRAModuleType.KEY,
            LoRAModuleType.VALUE,
            LoRAModuleType.QUERY,
        ],
        description="All Linear layers that will receive a parallel LoRa Adapter",
    )

    dropout: float = Field(default=0.0, description="The dropout probability for Lora layers.")

    alpha: int = Field(default=1, description="The alpha parameter for Lora scaling")

    bias: bool = Field(
        default=False,
        description="Boolean to decide to use Bias or not in LoRa Modules",
    )

    kaiming_a: float = Field(
        default=math.sqrt(5),
        description="Initializing the a for kaiming_init of the A matrix.",
    )
