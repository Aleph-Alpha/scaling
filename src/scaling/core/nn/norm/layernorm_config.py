from enum import Enum

from pydantic import Field

from scaling.core.config import BaseConfig


class LayerNormOptimizationType(Enum):
    TORCH = "torch"


class LayerNormConfig(BaseConfig):
    optimization_type: LayerNormOptimizationType = Field(
        LayerNormOptimizationType.TORCH,
        description="select an optimization type for the layer norm call, "
        "if anything other than torch is selected "
        "the optional gpu_optimization dependencies need to be installed",
    )

    layernorm_epsilon: float = Field(1.0e-5, description="A value added to the denominator for numerical stability")
