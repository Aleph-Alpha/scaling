from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from ..config import BaseConfig


class PipePartitionMethod(Enum):
    UNIFORM = "uniform"
    BALANCED = "balanced"


class ActivationCheckpointingType(Enum):
    EVERY_PIPE_STAGE = "every_pipe_stage"
    EVERY_LAYER = "every_layer"
    DISABLED = "disabled"


class TopologyConfig(BaseConfig):
    global_rank: Optional[int] = Field(
        None,
        description="",
        ge=0,
    )

    world_size: int = Field(
        description="",
        gt=0,
    )

    local_slot: Optional[int] = Field(
        None,
        description="",
        ge=0,
    )

    model_parallel_size: int = Field(
        description="",
        gt=0,
    )

    pipe_parallel_size: int = Field(
        description="",
        gt=0,
    )

    data_parallel_size: int = Field(
        description="",
        gt=0,
    )

    global_batch_size: int = Field(
        description="global train batch size including all gradient accumulation steps",
        gt=0,
    )

    micro_batch_size: int = Field(
        description="Batch size for one training micro step. "
        "This is used when the global_batch_size cannot fit in GPU memory "
        "to determine the number of gradient accumulation steps.",
        gt=0,
    )

    gradient_accumulation_steps: int = Field(
        description="Number of gradient accumulation. "
        "This is used when the global_batch_size cannot fit in GPU memory "
        "to determine the number of gradient accumulation steps.",
        gt=0,
    )

    pipe_partition_method: PipePartitionMethod = Field(
        PipePartitionMethod.UNIFORM,
        description="Method to assign layers to pipeline stages",
    )

    pipe_partition_overwrite: Optional[List[int]] = Field(
        None,
        description="manually set pipe partitions",
    )

    activation_checkpointing_type: ActivationCheckpointingType = Field(
        ActivationCheckpointingType.DISABLED,
        description="",
    )

    sequence_parallel: bool = Field(
        False,
        description="",
    )

    @model_validator(mode="before")
    def validate_parallelization_and_batch(cls, values: Dict[Any, Any]) -> Dict[Any, Any]:
        # get config values
        global_rank = values.get("global_rank")
        world_size = values.get("world_size")

        model_parallel_size = values.get("model_parallel_size")
        pipe_parallel_size = values.get("pipe_parallel_size")
        data_parallel_size = values.get("data_parallel_size")

        global_batch_size = values.get("global_batch_size")
        micro_batch_size = values.get("micro_batch_size")
        gradient_accumulation_steps = values.get("gradient_accumulation_steps")

        (
            model_parallel_size,
            pipe_parallel_size,
            data_parallel_size,
            world_size,
        ) = cls._calculate_parallelization_parameters(
            model_parallel_size=model_parallel_size,
            pipe_parallel_size=pipe_parallel_size,
            data_parallel_size=data_parallel_size,
            world_size=world_size,
        )

        global_batch_size, micro_batch_size, gradient_accumulation_steps = cls._calculate_batch_parameters(
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            data_parallel_size=data_parallel_size,
        )

        # overwrite values with updated information
        values["global_rank"] = global_rank
        values["world_size"] = world_size
        values["model_parallel_size"] = model_parallel_size
        values["pipe_parallel_size"] = pipe_parallel_size
        values["data_parallel_size"] = data_parallel_size
        values["global_batch_size"] = global_batch_size
        values["gradient_accumulation_steps"] = gradient_accumulation_steps
        values["micro_batch_size"] = micro_batch_size

        return values

    @staticmethod
    def _calculate_parallelization_parameters(
        model_parallel_size: Optional[int],
        pipe_parallel_size: Optional[int],
        data_parallel_size: Optional[int],
        world_size: Optional[int],
    ) -> tuple[int, int, int, int]:
        parameters = [
            model_parallel_size,
            pipe_parallel_size,
            data_parallel_size,
            world_size,
        ]
        set_parameter_count = len([1 for element in parameters if element is not None])

        if set_parameter_count < 3:
            raise AssertionError(
                "At least 3 out of 4 parallelization parameters (model_parallel_size, pipe_parallel_size, "
                "data_parallel_size and world_size) need to be set."
            )

        if world_size is None:
            world_size = model_parallel_size * pipe_parallel_size * data_parallel_size  # type: ignore[operator]
        if model_parallel_size is None:
            model_parallel_size = world_size // (pipe_parallel_size * data_parallel_size)  # type: ignore[operator]
        if pipe_parallel_size is None:
            pipe_parallel_size = world_size // (model_parallel_size * data_parallel_size)  # type: ignore[operator]
        if data_parallel_size is None:
            data_parallel_size = world_size // (model_parallel_size * pipe_parallel_size)  # type: ignore[operator]

        return model_parallel_size, pipe_parallel_size, data_parallel_size, world_size

    @staticmethod
    def _calculate_batch_parameters(
        global_batch_size: Optional[int],
        micro_batch_size: Optional[int],
        gradient_accumulation_steps: Optional[int],
        data_parallel_size: int,
    ) -> tuple[int, int, int]:
        parameters = [global_batch_size, micro_batch_size, gradient_accumulation_steps]
        set_parameter_count = len([1 for element in parameters if element is not None])

        if set_parameter_count < 2:
            raise AssertionError(
                "At least 2 out of 3 batch size parameters (global_batch_size, micro_batch_size, "
                "and gradient_accumulation_steps) need to be set."
            )

        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = global_batch_size // (micro_batch_size * data_parallel_size)  # type: ignore[operator]

        if micro_batch_size is None:
            micro_batch_size = global_batch_size // (gradient_accumulation_steps * data_parallel_size)  # type: ignore[operator]

        if global_batch_size is None:
            global_batch_size = micro_batch_size * gradient_accumulation_steps * data_parallel_size

        # make sure config is consistent if all values are set
        assert global_batch_size == (micro_batch_size * gradient_accumulation_steps * data_parallel_size), (
            f"global_batch_size {global_batch_size} "
            f"does not equal the product of micro_batch_size ({micro_batch_size}) "
            f"and gradient_accumulation_steps ({gradient_accumulation_steps}) "
            f"and data_parallel_size ({data_parallel_size})."
        )

        return (
            global_batch_size,
            micro_batch_size,
            gradient_accumulation_steps,
        )
