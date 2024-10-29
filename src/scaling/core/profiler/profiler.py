import collections
import json
from contextlib import contextmanager
from typing import Generator, Mapping, NamedTuple, Optional

import torch

from scaling.core.profiler.profiler_config import ProfilerConfig
from scaling.core.profiler.timer import SynchronizedTimer
from scaling.core.topology import Topology


class ProfilerObservation(NamedTuple):
    timer_name: str
    step: int
    micro_batch_id: int
    buffer_id: int
    pipe_parallel_rank: int
    data_parallel_rank: int
    model_parallel_rank: int
    duration: float


class Profiler:
    def __init__(self, config: ProfilerConfig, topology: Topology) -> None:
        self.config = config
        self.topology = topology

        # local step counter and variables
        self.steps = 0
        self.step_save = 0
        self.end_step = self.config.profile_start_at_step + self.config.profile_steps
        self.enabled = self.config.profile_steps > 0 and self.config.profiler_output is not None

        # all timed steps
        self.observations: list[ProfilerObservation] = []

    def step(self) -> None:
        """
        initialize a step for the profiler
        this function needs to be called before recording timings
        """
        self.steps += 1
        self.timers: Mapping[str, Mapping[Optional[int], Mapping[Optional[int], SynchronizedTimer]]] = (
            collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: SynchronizedTimer()))
            )
        )

    def flush(self) -> None:
        """
        records current step timers and
        potentially saves results out to disc
        """

        if not self.enabled:
            return

        for timer_name, micro_batch_data in self.timers.items():
            for micro_batch_id, buffer_data in micro_batch_data.items():
                for buffer_id, timer in buffer_data.items():
                    self.observations.append(
                        ProfilerObservation(
                            timer_name=timer_name,
                            step=self.step_save,
                            micro_batch_id=micro_batch_id,  # type: ignore[arg-type]
                            buffer_id=buffer_id,  # type: ignore[arg-type]
                            pipe_parallel_rank=self.topology.pipe_parallel_rank,
                            data_parallel_rank=self.topology.data_parallel_rank,
                            model_parallel_rank=self.topology.model_parallel_rank,
                            duration=timer.duration(),
                        )
                    )

        self.step_save += 1
        if self.enabled and self.steps == (self.end_step):
            self.save()

    def save(self) -> None:
        if self.config.profiler_output is None:
            return

        observations_per_rank: list = [None for _ in range(self.topology.config.world_size)]
        torch.distributed.gather_object(
            self.observations,
            observations_per_rank if self.topology.config.global_rank == 0 else None,
            dst=0,
        )

        if self.topology.config.global_rank == 0:
            assert observations_per_rank is not None
            observations = [obs for obs_list in observations_per_rank for obs in obs_list]
            self.config.profiler_output.parent.mkdir(exist_ok=True, parents=True)
            json.dump(
                {
                    "pipe_parallel_size": self.topology.config.pipe_parallel_size,
                    "data_parallel_size": self.topology.config.data_parallel_size,
                    "model_parallel_size": self.topology.config.model_parallel_size,
                    "gradient_accumulation_steps": self.topology.config.gradient_accumulation_steps,
                    "observations": [o._asdict() for o in observations],
                },
                open(self.config.profiler_output, "w", encoding="UTF-8"),
                indent=4,
            )

    def start_timer(
        self,
        timer_name: str,
        micro_batch_id: Optional[int],
        buffer_id: Optional[int],
    ) -> None:
        # return if not enabled outside of to be recorded steps
        if not self.enabled or (self.steps < self.config.profile_start_at_step or self.steps > self.end_step):
            return

        self.timers[timer_name][micro_batch_id][buffer_id].start()

    def stop_timer(
        self,
        timer_name: str,
        micro_batch_id: Optional[int],
        buffer_id: Optional[int],
    ) -> None:
        # return if not enabled outside of to be recorded steps
        if not self.enabled or (self.steps < self.config.profile_start_at_step or self.steps > self.end_step):
            return

        self.timers[timer_name][micro_batch_id][buffer_id].stop()

    @contextmanager
    def time(
        self,
        timer_name: str,
        micro_batch_id: Optional[int],
        buffer_id: Optional[int],
    ) -> Generator[None, None, None]:
        self.start_timer(timer_name, micro_batch_id, buffer_id)
        yield
        self.stop_timer(timer_name, micro_batch_id, buffer_id)
