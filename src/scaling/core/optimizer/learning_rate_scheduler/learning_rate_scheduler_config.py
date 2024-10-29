from enum import Enum

from pydantic import Field

from scaling.core.config import BaseConfig


class LearningRateDecayStyle(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class LearningRateSchedulerConfig(BaseConfig):
    learning_rate: float = Field(
        0.0,
        description="Base learning rate; this is also the maximum learning rate.",
    )

    learning_rate_minimum: float = Field(
        0.0,
        description="Minimum learning rate below which a step's learning rate will never drop. "
        "This is the final learning rate after the schedule has been applied.",
    )

    learning_rate_decay_style: LearningRateDecayStyle = Field(
        LearningRateDecayStyle.COSINE,
        description="Shape of the learning rate decay after warm up",
    )

    learning_rate_decay_iters: int = Field(
        0,
        description="Number of iterations within which the learning rate follows the schedule. "
        "Warmup iterations are included.",
    )

    learning_rate_warmup_steps: int = Field(
        0,
        description="Number of warmup steps during which the learning rate "
        "is linearly increased to the maximum learning rate. "
        "The actual schedule starts after the warmup steps.",
    )

    learning_rate_warmup_delay_steps: int = Field(
        0,
        description="Number of steps for with the learning rate remains zero, until the actual warmup starts.",
    )
