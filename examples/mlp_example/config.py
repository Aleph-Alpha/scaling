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

from scaling.core.logging import LoggerConfig


class MLPArchitectureConfig(BaseConfig):
    n_hidden_layers: int = Field(
        default=0,
        ge=0, description=(
            "The number of layers in the network, excluding input and "
            "output layers."
        )
    )
    hidden_dim: int = Field(
        default=64,
        gt=0, description=(
            "The number of hidden units in each hidden layer."
        )
    )


class TrainingConfig(BaseConfig):
    weight_decay: float = Field(0.0001, description="")


class MLPConfig(BaseConfig):
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

    architecture: MLPArchitectureConfig = Field(MLPArchitectureConfig(), description="")
