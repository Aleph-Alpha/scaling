from enum import Enum
from typing import Optional

import torch
from pydantic import Field

from scaling.core import (
    BaseConfig,
    LearningRateSchedulerConfig,
    OptimizerConfig,
    ProfilerConfig,
    RunnerConfig,
    TopologyConfig,
    TrainerConfig,
)
from scaling.core.logging.logger_config import LoggerConfig


class Precision(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def dtype(self) -> torch.dtype:
        if self == Precision.FLOAT16:
            return torch.float16
        elif self == Precision.BFLOAT16:
            return torch.bfloat16
        elif self == Precision.FLOAT32:
            return torch.float32
        else:
            raise NotImplementedError


class TrainingConfig(BaseConfig):
    weight_decay: float = Field(0.0001, description="")
    precision: Precision = Field(Precision.FLOAT32, description="")
    weight_tying: bool = Field(False, description="")
    bitfit_bias_name: Optional[str] = Field(None, description="")


class MinimalConfig(BaseConfig):
    runner: RunnerConfig = Field(
        RunnerConfig(),
        description="",
    )

    logger: LoggerConfig = Field(
        LoggerConfig(),
        description="",
    )

    topology: TopologyConfig = Field(
        TopologyConfig(  # type: ignore[call-arg]
            model_parallel_size=1,
            pipe_parallel_size=1,
            data_parallel_size=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        ),
        description="",
    )
    optimizer: OptimizerConfig = Field(OptimizerConfig(), description="")

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(LearningRateSchedulerConfig(), description="")

    training: TrainingConfig = Field(TrainingConfig(), description="")

    trainer: TrainerConfig = Field(TrainerConfig(), description="")

    profiler: ProfilerConfig = Field(ProfilerConfig(), description="")
