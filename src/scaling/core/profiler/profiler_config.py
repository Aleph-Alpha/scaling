from pathlib import Path
from typing import Optional

from pydantic import Field

from ..config import BaseConfig


class ProfilerConfig(BaseConfig):
    profile_steps: int = Field(0, description="number of to be timed steps, will not run profiling if set to 0")

    profile_start_at_step: int = Field(
        10,
        description="start of profiler after this many steps of the current process. "
        "Not starting at step 0 give the GPUs time to (physically) warm up"
        " and only starts timing after initial meta data has been synced",
    )

    profiler_output: Optional[Path] = Field(
        None,
        description="start of profiler after this many steps of the current process. "
        "Not starting at step 0 give the GPUs time to (physically) warm up "
        "and only starts timing after initial meta data has been synced",
    )
