from typing import Optional

from pydantic import Field

from scaling.core.config import BaseConfig
from scaling.core.optimizer.learning_rate_scheduler import LearningRateSchedulerConfig


class OptimizerParamGroupConfig(BaseConfig):
    name: Optional[str] = Field(
        None,
        description="Name of the parameter group for logging",
    )

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(
        LearningRateSchedulerConfig(),
        description="Configuration of the parameter group's learning rate schedule",
    )

    weight_decay: float = Field(
        1e-2,
        description="Weight decay for all parameters within the parameter group",
    )

    independent_weight_decay: bool = Field(
        False,
        description="""If True, the optimizer update contribution from the weight decay
         will only be multiplied by the weight decay factor and not additionally by the learning rate.""",
    )
