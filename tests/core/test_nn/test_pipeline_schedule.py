from pathlib import Path

import pytest
from PIL import Image  # type: ignore

from scaling.core import (
    PipelineScheduleInference,
    PipelineScheduleTrain,
)


@pytest.mark.parametrize("pipe_parallel_size", [1, 2, 7, 16, 32])
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2, 7, 16, 32])
def test_pipeline_train_schedule(
    tmp_path: Path,
    pipe_parallel_size: int,
    gradient_accumulation_steps: int,
):
    # visualize() iterates over pipe parallel ranks.
    # There is no need in the test case to do so

    schedule_image = PipelineScheduleTrain.visualize(
        gradient_accumulation_steps=gradient_accumulation_steps,
        pipe_parallel_size=pipe_parallel_size,
    )

    assert isinstance(schedule_image, Image.Image)
    image_path = tmp_path / (
        f"train_schedule_pipe_parallel_size_{pipe_parallel_size}_"
        f"gradient_accumulation_steps_{gradient_accumulation_steps}.png"
    )
    schedule_image.save(str(image_path))


def test_visualize_train_profile(tmp_path: Path):
    profile_file = Path(__file__).parent.absolute() / "profile.json"
    timings, image = PipelineScheduleTrain.visualize_profile(
        profile_file=profile_file,
        milliseconds_per_pixel=0.0001,  # very small due to the very small test model and timings
        pipe_pixels=200,
    )
    image.save(str(tmp_path / "profile.png"))


@pytest.mark.parametrize("pipe_parallel_size", [1, 2, 7, 16, 32])
def test_pipeline_inference_schedule(tmp_path: Path, pipe_parallel_size: int):
    # visualize() iterates over pipe parallel ranks.
    # There is no need in the test case to do so
    schedule_image = PipelineScheduleInference.visualize(
        gradient_accumulation_steps=1,
        pipe_parallel_size=pipe_parallel_size,
    )
    assert isinstance(schedule_image, Image.Image)
    schedule_image.save(str(tmp_path / f"inference_schedule_{pipe_parallel_size}_gradient_accumulation_steps_1.png"))

    # the inference schedule should be independent of gradient accumulation steps
    schedule_image_2 = PipelineScheduleInference.visualize(
        gradient_accumulation_steps=2,
        pipe_parallel_size=pipe_parallel_size,
    )
    schedule_image_2.save(str(tmp_path / f"inference_schedule_{pipe_parallel_size}_gradient_accumulation_steps_2.png"))
