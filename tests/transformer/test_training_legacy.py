from pathlib import Path

import pytest

from .test_training import execute_run_training


@pytest.mark.training_legacy
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 2, 4),
        (2, 2, 4),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(False, "float32")])
@pytest.mark.parametrize("legacy_dataset", [True])
@pytest.mark.parametrize("use_determined", [False])
@pytest.mark.parametrize("weight_tying", [True])
@pytest.mark.parametrize("kernel", ["torch"])
@pytest.mark.parametrize("sequence_parallel", [False])
def test_transformer_training_legacy(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    legacy_dataset: bool,
    use_determined: bool,
    weight_tying: bool,
    kernel: str,
    sequence_parallel: bool,
):
    execute_run_training(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        gradient_accumulation_steps,
        enable_loss_scaling,
        precision,
        legacy_dataset,
        use_determined,
        weight_tying,
        False,
        kernel,
        sequence_parallel,
    )
