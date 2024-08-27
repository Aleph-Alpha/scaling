from typing import Optional

from pydantic import Field

from ..config import BaseConfig
from .learning_rate_scheduler import LearningRateSchedulerConfig


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
