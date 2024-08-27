from pydantic import Field

from scaling.core.config import BaseConfig


class RotaryConfig(BaseConfig):
    dimensions: int = Field(0, description="")

    base: int = Field(10000, description="")

    max_seq_length: int = Field(2048, description="")
