from pydantic import Field

from scaling.core.config import BaseConfig
from scaling.core.optimizer.loss_scaler import LossScalerConfig


class OptimizerConfig(BaseConfig):
    method: str = Field("adamw", description="Which optimization method to use.")

    beta1: float = Field(
        0.9,
        description="First coefficient used for computing running averages of gradient and its square",
    )

    beta2: float = Field(
        0.95,
        description="Second coefficient used for computing running averages of gradient and its square",
    )

    eps: float = Field(
        1e-8,
        description="term added to the denominator to improve numerical stability (default: 1e-8)",
    )

    gradient_clipping: float = Field(1.0, description="clip global l2 grads to this value, deactivate if 0.0", ge=0.0)

    allreduce_bucket_size: int = Field(500000000, description="number of floating points to allreduce in one go", gt=0)

    loss_scaler: LossScalerConfig = Field(
        LossScalerConfig(),
        description="Configuration of the loss scaler",
    )

    zero: bool = Field(False, description="enable zero stage 1 optimizer")

    zero_save_static: bool = Field(
        False,
        description="Save zero state dict without merging parameters and optimizer states. "
        "This may be used in large scale trainings to save and load checkpoints faster and not run oom.",
    )

    debug_log: bool = Field(False)
