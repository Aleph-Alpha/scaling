import collections
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scaling.core.nn.pipeline_schedule.instructions import InstructionBase
from scaling.core.topology import Topology, TopologyConfig

DEPENDENCY_MAP = {
    "InstructionRecvActivation": {
        "instruction": "InstructionSendActivation",
        "previous": True,
    },
    "InstructionSendActivation": {
        "instruction": "InstructionRecvActivation",
        "previous": False,
    },
    "InstructionRecvGrad": {"instruction": "InstructionSendGrad", "previous": False},
    "InstructionSendGrad": {"instruction": "InstructionRecvGrad", "previous": True},
}


class PipelineScheduleBase(ABC):
    def __init__(self, topology: Topology):
        self.topology = topology

    @abstractmethod
    def instructions(self) -> list[InstructionBase]:
        pass

    @abstractmethod
    def required_buffer_count(self) -> int:
        pass

    @classmethod
    def illustrate(cls, gradient_accumulation_steps: int, pipe_parallel_size: int) -> dict[str, Any]:
        # get all instructions
        next_commands_by_rank = collections.defaultdict(list)
        for pipe_parallel_rank in range(pipe_parallel_size):
            topology = Topology(
                config=TopologyConfig(  # type: ignore[call-arg]
                    global_rank=pipe_parallel_rank,
                    # in this case with mp 1 and pp1 the pipe parallel rank will be the same as the global rank
                    pipe_parallel_size=pipe_parallel_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    model_parallel_size=1,
                    data_parallel_size=1,
                    micro_batch_size=1,
                )
            )
            pipe_schedule = cls(topology=topology)
            for instruction in pipe_schedule.instructions():
                next_commands_by_rank[pipe_parallel_rank].append(instruction)

        # reverse instructions to read from right / be able to pop
        for pipe_parallel_rank in range(pipe_parallel_size):
            next_commands_by_rank[pipe_parallel_rank] = list(reversed(next_commands_by_rank[pipe_parallel_rank]))

        # layout
        steps = list()
        next_command_by_rank = {
            pipe_parallel_rank: next_commands_by_rank[pipe_parallel_rank].pop()
            for pipe_parallel_rank in range(pipe_parallel_size)
        }
        while any(len(commands) > 0 for commands in next_commands_by_rank.values()) or not all(
            command is None for command in next_command_by_rank.values()
        ):
            # check for each rank whether dependencies are fulfilled
            dependencies_fulfilled = {pipe_parallel_rank: False for pipe_parallel_rank in range(pipe_parallel_size)}
            for pipe_parallel_rank in range(pipe_parallel_size):
                if next_command_by_rank[pipe_parallel_rank] is not None:
                    if next_command_by_rank[pipe_parallel_rank].name == "ReduceTiedGrads":
                        if not (
                            all(
                                next_command_by_rank[pipe_parallel_rank] is not None
                                for pipe_parallel_rank in range(pipe_parallel_size)
                            )
                            and all(
                                next_command_by_rank[pipe_parallel_rank].name == "ReduceTiedGrads"
                                for pipe_parallel_rank in range(pipe_parallel_size)
                            )
                        ):
                            continue

                    elif next_command_by_rank[pipe_parallel_rank].name in DEPENDENCY_MAP:
                        # check whether command can be executed

                        dependency_rank = (
                            (pipe_parallel_rank - 1)
                            if DEPENDENCY_MAP[next_command_by_rank[pipe_parallel_rank].name]["previous"]
                            else (pipe_parallel_rank + 1)
                        )

                        if dependency_rank < 0:
                            dependency_rank = pipe_parallel_size - 1
                        if dependency_rank > pipe_parallel_size - 1:
                            dependency_rank = 0

                        dependency_instruction = DEPENDENCY_MAP[next_command_by_rank[pipe_parallel_rank].name][
                            "instruction"
                        ]
                        if next_command_by_rank[dependency_rank].name != dependency_instruction:
                            continue  # dependency fulfilled is by default false
                    dependencies_fulfilled[pipe_parallel_rank] = True

            # schedule all steps for which dependencies are fulfilled
            step_commands: dict[int, Optional[dict[str, Any]]] = {
                pipe_parallel_rank: None for pipe_parallel_rank in range(pipe_parallel_size)
            }
            for (
                pipe_parallel_rank,
                dependency_fulfilled,
            ) in dependencies_fulfilled.items():
                if not dependency_fulfilled:
                    continue
                step_commands[pipe_parallel_rank] = {
                    "name": next_command_by_rank[pipe_parallel_rank].name,
                }

                try:
                    next_command_by_rank[pipe_parallel_rank] = next_commands_by_rank[pipe_parallel_rank].pop()
                except IndexError:
                    next_command_by_rank[pipe_parallel_rank] = None  # type: ignore[assignment]

            steps.append(step_commands)

        # count for each rank how many time steps are idling
        count_idling = {pipe_parallel_rank: 0 for pipe_parallel_rank in range(pipe_parallel_size)}
        count_idling_total = 0
        for step in steps:
            for pipe_parallel_rank, command in step.items():
                if command is None:
                    count_idling[pipe_parallel_rank] += 1
                    count_idling_total += 1

        return {
            "steps": steps,
            "count_idling": count_idling,
            "pct_idling": {rank: (cnt / len(steps)) for rank, cnt in count_idling.items()},
            "count_idling_total": count_idling_total,
            "pct_idling_total": count_idling_total / (len(steps) * pipe_parallel_size),
        }

    @classmethod
    def visualize(cls, gradient_accumulation_steps: int, pipe_parallel_size: int) -> Image.Image:
        # get illustration
        illustration = cls.illustrate(
            gradient_accumulation_steps=gradient_accumulation_steps,
            pipe_parallel_size=pipe_parallel_size,
        )

        step_pixels = 50
        pipe_pixels = 100
        dpi = 100  # noqa: F841

        # prepare visualization data
        step_names = [str(i) for i in range(len(illustration["steps"]))]
        pipe_parallel_names = [f"pp_{i}" for i in range(pipe_parallel_size)]

        heat = np.zeros(
            (
                len(step_names) * step_pixels,
                len(pipe_parallel_names) * pipe_pixels,
                3,
            ),
            dtype=np.uint8,
        )

        for step_index in range(len(illustration["steps"])):
            for pipe_parallel_rank in range(pipe_parallel_size):
                if (
                    illustration["steps"][step_index][pipe_parallel_rank] is not None
                    and illustration["steps"][step_index][pipe_parallel_rank]["name"] is not None
                ):
                    name = illustration["steps"][step_index][pipe_parallel_rank]["name"]
                    name = name.replace("Instruction", "")
                    if name in ["ForwardPass", "BackwardPass"]:
                        color = np.array([34, 139, 34], dtype=np.uint8)
                    elif name in [
                        "SendActivation",
                        "RecvActivation",
                        "SendGrad",
                        "RecvGrad",
                    ]:
                        color = np.array([255, 255, 0], dtype=np.uint8)
                    elif name in ["LoadMicroBatch"]:
                        color = np.array([0, 0, 255], dtype=np.uint8)
                    else:
                        color = np.array([191, 191, 191], dtype=np.uint8)
                else:
                    color = np.array([255, 255, 255], dtype=np.uint8)
                heat[
                    (step_index * step_pixels + 1) : (step_index * step_pixels + step_pixels - 1),
                    (pipe_parallel_rank * pipe_pixels + 1) : (pipe_parallel_rank * pipe_pixels + pipe_pixels - 1),
                ] = color

        # visualize
        image = Image.fromarray(heat).convert("RGB")
        image_draw = ImageDraw.Draw(image)
        image_font = ImageFont.load_default()

        for step_index, step in enumerate(illustration["steps"]):
            for pipe_parallel_rank, command in step.items():
                if command is not None:
                    image_draw.text(
                        (
                            (pipe_parallel_rank * pipe_pixels + 2),
                            (step_index * step_pixels + 2),
                        ),
                        command["name"].replace("Instruction", ""),
                        font=image_font,
                        fill=(0, 0, 0),
                    )
        return image

    @classmethod
    def visualize_profile(
        cls,
        profile_file: Path,
        milliseconds_per_pixel: float = 10.0,
        pipe_pixels: int = 200,
    ) -> tuple[dict[str, Any], Image.Image]:
        profile_data = json.load(open(profile_file, "r", encoding="UTF-8"))

        simulation_engine = SimulationEngine(
            schedule=cls,
            profile_data=profile_data,
        )

        # simulate
        simulation_engine.simulate()
        result = simulation_engine.summarize()
        image = simulation_engine.visualize(milliseconds_per_pixel=milliseconds_per_pixel, pipe_pixels=pipe_pixels)
        return result, image

    def _valid_micro_batch(self, micro_batch_id: int) -> bool:
        return 0 <= micro_batch_id < self.topology.config.gradient_accumulation_steps

    def _is_valid_pipe_parallel_rank(self, pipe_parallel_rank: Optional[int]) -> bool:
        return pipe_parallel_rank is not None and 0 <= pipe_parallel_rank < self.topology.config.pipe_parallel_size


class SimulationStatus(Enum):
    INITIAL = "INITIAL"
    DONE = "DONE"


class SimulationInstruction:
    def __init__(
        self,
        name: str,
        micro_batch_id: int,
        buffer_id: int,
        status: SimulationStatus,
        time_duration: float,
        time_start: float,
        time_end: float,
    ) -> None:
        self.name = name
        self.micro_batch_id = micro_batch_id
        self.buffer_id = buffer_id
        self.status = status
        self.time_duration = time_duration
        self.time_start = time_start
        self.time_end = time_end

    def __repr__(self) -> str:
        return f"{self.name} @ micro_batch {self.micro_batch_id}"


class SimulationEngine:
    INSTRUCTION_TIMER_MAPPING = {
        "InstructionLoadMicroBatch": "BATCH_INPUT",
        "InstructionForwardPass": "FORWARD",
        "InstructionSendActivation": "FORWARD_SEND_ACTIVATIONS",
        "InstructionRecvActivation": "FORWARD_RECV_ACTIVATIONS",
        "InstructionBackwardPass": "BACKWARD",
        "InstructionSendGrad": "BACKWARD_SEND_GRADS",
        "InstructionRecvGrad": "BACKWARD_RECV_GRADS",
        "InstructionReduceTiedGrads": "REDUCE_TIED_GRADS",
        "InstructionOptimizerStep": "OPTIMIZER_STEP",
    }

    def __init__(
        self,
        schedule: type[PipelineScheduleBase],
        profile_data: dict,
    ) -> None:
        self.schedule = schedule
        self.profile_data = profile_data

        # topology information
        self.gradient_accumulation_steps = profile_data["gradient_accumulation_steps"]
        self.pipe_parallel_size = profile_data["pipe_parallel_size"]

        self.simulation_instructions_by_pipe_parallel_rank: dict[int, list[SimulationInstruction]] = dict()

        self.collect_simulation_instructions()

    def collect_simulation_instructions(self) -> None:
        """
        Initializes a schedule and collects all simulation instructions for all pipe parallel ranks
        """
        for pipe_parallel_rank in range(self.pipe_parallel_size):
            topology = Topology(
                config=TopologyConfig(  # type: ignore[call-arg]
                    global_rank=pipe_parallel_rank,
                    # in this case with mp 1 and pp1 the pipe parallel rank will be the same as the global rank
                    pipe_parallel_size=self.pipe_parallel_size,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    model_parallel_size=1,
                    data_parallel_size=1,
                    micro_batch_size=1,
                )
            )
            pipe_schedule = self.schedule(topology=topology)
            for instruction in pipe_schedule.instructions():
                if pipe_parallel_rank not in self.simulation_instructions_by_pipe_parallel_rank:
                    self.simulation_instructions_by_pipe_parallel_rank[pipe_parallel_rank] = list()
                self.simulation_instructions_by_pipe_parallel_rank[pipe_parallel_rank].append(
                    SimulationInstruction(
                        name=instruction.name,
                        micro_batch_id=instruction.micro_batch_id,  # type: ignore[arg-type]
                        buffer_id=instruction.buffer_id,  # type: ignore[arg-type]
                        status=SimulationStatus.INITIAL,
                        time_duration=self.get_simulation_time_duration(
                            pipe_parallel_rank=pipe_parallel_rank,
                            name=instruction.name,
                            micro_batch_id=instruction.micro_batch_id,
                            buffer_id=instruction.buffer_id,
                        ),
                        time_start=None,  # type: ignore[arg-type]
                        time_end=None,  # type: ignore[arg-type]
                    )
                )

    def get_duration(
        self, pipe_parallel_rank: int, name: str, micro_batch_id: Optional[int], buffer_id: Optional[int]
    ) -> float:
        if micro_batch_id is None:
            micro_batch_id = -1
        if buffer_id is None:
            buffer_id = -1
        # get relevant observations
        # due to the allreduce in the profiler there are irrelevant items
        observations = [
            observation
            for observation in self.profile_data["observations"]
            if observation["pipe_parallel_rank"] == pipe_parallel_rank
            and observation["micro_batch_id"] == micro_batch_id
            and observation["buffer_id"] == buffer_id
            and observation["timer_name"] == name.replace("Instruction", "")
        ]
        assert len(observations) > 0

        # get mean duration
        duration = sum([o["duration"] for o in observations]) / len(observations)
        return duration

    def get_simulation_time_duration(
        self, pipe_parallel_rank: int, name: str, micro_batch_id: Optional[int], buffer_id: Optional[int]
    ) -> float:
        duration = self.get_duration(
            pipe_parallel_rank=pipe_parallel_rank,
            name=name,
            micro_batch_id=micro_batch_id,
            buffer_id=buffer_id,
        )

        # in case of communications timers potentially include wait times for other ranks
        # these wait times are eliminated
        if name == "InstructionSendActivation":
            duration = min(
                duration,
                self.get_duration(
                    pipe_parallel_rank=pipe_parallel_rank + 1,
                    name="InstructionRecvActivation",
                    micro_batch_id=micro_batch_id,
                    buffer_id=buffer_id,
                ),
            )
        elif name == "InstructionRecvActivation":
            duration = min(
                duration,
                self.get_duration(
                    pipe_parallel_rank=pipe_parallel_rank - 1,
                    name="InstructionSendActivation",
                    micro_batch_id=micro_batch_id,
                    buffer_id=buffer_id,
                ),
            )
        elif name == "InstructionSendGrad":
            duration = min(
                duration,
                self.get_duration(
                    pipe_parallel_rank=pipe_parallel_rank - 1,
                    name="InstructionRecvGrad",
                    micro_batch_id=micro_batch_id,
                    buffer_id=buffer_id,
                ),
            )
        elif name == "InstructionRecvGrad":
            duration = min(
                duration,
                self.get_duration(
                    pipe_parallel_rank=pipe_parallel_rank + 1,
                    name="InstructionSendGrad",
                    micro_batch_id=micro_batch_id,
                    buffer_id=buffer_id,
                ),
            )

        return duration

    def reset_simulation(self) -> None:
        for simulation_instructions in self.simulation_instructions_by_pipe_parallel_rank.values():
            for simulation_instruction in simulation_instructions:
                simulation_instruction.status = SimulationStatus.INITIAL
                simulation_instruction.time_start = None  # type: ignore[assignment]
                simulation_instruction.time_end = None  # type: ignore[assignment]

    def simulate(self) -> None:
        """
        - we iterate until each pipeline stage has no next commands to run
        - a pipeline stage schedules next instructions whenever ready in case of no dependencies
        - a pipeline stage waits for other pipeline stages in case of dependencies
        - steps are not discrete as they may overlap in time
        - the main task of the simulation is to add a start and end time to each instruction while aligning dependencies
        """
        self.reset_simulation()
        time_spent_pipeline_stage = {
            pipe_parallel_rank: 0.0 for pipe_parallel_rank in self.simulation_instructions_by_pipe_parallel_rank.keys()
        }

        simulation_iteration = 0
        next_instructions_for_ranks = {
            pipe_parallel_rank: self.next_instruction_for_rank(pipe_parallel_rank=pipe_parallel_rank)
            for pipe_parallel_rank in self.simulation_instructions_by_pipe_parallel_rank.keys()
        }
        while not all([i is None for i in next_instructions_for_ranks.values()]):
            simulation_iteration += 1

            # set start time for dependent jobs
            for pipe_parallel_rank in self.simulation_instructions_by_pipe_parallel_rank.keys():
                next_instruction = next_instructions_for_ranks[pipe_parallel_rank]
                if next_instruction is None:
                    continue
                if next_instruction.name == "InstructionSendActivation":
                    next_instruction_next_rank = next_instructions_for_ranks[pipe_parallel_rank + 1]
                    assert next_instruction_next_rank is not None
                    if next_instruction_next_rank.name == "InstructionRecvActivation":
                        time_start = max(
                            time_spent_pipeline_stage[pipe_parallel_rank],
                            time_spent_pipeline_stage[pipe_parallel_rank + 1],
                        )
                        time_spent_pipeline_stage[pipe_parallel_rank] = time_start
                        time_spent_pipeline_stage[pipe_parallel_rank + 1] = time_start
                elif next_instruction.name == "InstructionRecvActivation":
                    next_instruction_prev_rank = next_instructions_for_ranks[pipe_parallel_rank - 1]
                    assert next_instruction_prev_rank is not None
                    if next_instruction_prev_rank.name == "InstructionSendActivation":
                        time_start = max(
                            time_spent_pipeline_stage[pipe_parallel_rank],
                            time_spent_pipeline_stage[pipe_parallel_rank - 1],
                        )
                        time_spent_pipeline_stage[pipe_parallel_rank] = time_start
                        time_spent_pipeline_stage[pipe_parallel_rank - 1] = time_start
                elif next_instruction.name == "InstructionSendGrad":
                    next_instruction_prev_rank = next_instructions_for_ranks[pipe_parallel_rank - 1]
                    assert next_instruction_prev_rank is not None
                    if next_instruction_prev_rank.name == "InstructionRecvGrad":
                        time_start = max(
                            time_spent_pipeline_stage[pipe_parallel_rank],
                            time_spent_pipeline_stage[pipe_parallel_rank - 1],
                        )
                        time_spent_pipeline_stage[pipe_parallel_rank] = time_start
                        time_spent_pipeline_stage[pipe_parallel_rank - 1] = time_start
                elif next_instruction.name == "InstructionRecvGrad":
                    next_instruction_next_rank = next_instructions_for_ranks[pipe_parallel_rank + 1]
                    assert next_instruction_next_rank is not None
                    if next_instruction_next_rank.name == "InstructionSendGrad":
                        time_start = max(
                            time_spent_pipeline_stage[pipe_parallel_rank],
                            time_spent_pipeline_stage[pipe_parallel_rank + 1],
                        )
                        time_spent_pipeline_stage[pipe_parallel_rank] = time_start
                        time_spent_pipeline_stage[pipe_parallel_rank + 1] = time_start
                elif next_instruction.name == "InstructionReduceTiedGrads":
                    if all([i.name == "InstructionReduceTiedGrads" for i in next_instructions_for_ranks.values()]):  # type: ignore[union-attr]
                        time_start = max(time_spent_pipeline_stage.values())
                        for k in list(time_spent_pipeline_stage.keys()):
                            time_spent_pipeline_stage[k] = time_start
                elif next_instruction.name == "InstructionOptimizerStep":
                    if all([i.name == "InstructionOptimizerStep" for i in next_instructions_for_ranks.values()]):  # type: ignore[union-attr]
                        time_start = max(time_spent_pipeline_stage.values())
                        for k in list(time_spent_pipeline_stage.keys()):
                            time_spent_pipeline_stage[k] = time_start
                elif next_instruction.name in [
                    "InstructionLoadMicroBatch",
                    "InstructionForwardPass",
                    "InstructionBackwardPass",
                    "InstructionLoss",
                ]:
                    pass  # independent instructions
                else:
                    raise NotImplementedError

            # actually do the tasks
            for pipe_parallel_rank in self.simulation_instructions_by_pipe_parallel_rank.keys():
                next_instruction = next_instructions_for_ranks[pipe_parallel_rank]
                if next_instruction is None:
                    continue

                # make sure tasks are only done when in order
                # otherwise the start time above cannot be set
                if next_instruction.name == "InstructionSendActivation":
                    next_instruction_next_rank = next_instructions_for_ranks[pipe_parallel_rank + 1]
                    assert next_instruction_next_rank is not None
                    if next_instruction_next_rank.name != "InstructionRecvActivation":
                        continue
                elif next_instruction.name == "InstructionRecvActivation":
                    next_instruction_prev_rank = next_instructions_for_ranks[pipe_parallel_rank - 1]
                    assert next_instruction_prev_rank is not None
                    if next_instruction_prev_rank.name != "InstructionSendActivation":
                        continue
                elif next_instruction.name == "InstructionSendGrad":
                    next_instruction_prev_rank = next_instructions_for_ranks[pipe_parallel_rank - 1]
                    assert next_instruction_prev_rank is not None
                    if next_instruction_prev_rank.name != "InstructionRecvGrad":
                        continue
                elif next_instruction.name == "InstructionRecvGrad":
                    next_instruction_next_rank = next_instructions_for_ranks[pipe_parallel_rank + 1]
                    assert next_instruction_next_rank is not None
                    if next_instruction_next_rank.name != "InstructionSendGrad":
                        continue
                elif next_instruction.name == "InstructionReduceTiedGrads":
                    if not all([i.name == "InstructionReduceTiedGrads" for i in next_instructions_for_ranks.values()]):  # type: ignore[union-attr]
                        continue
                elif next_instruction.name == "InstructionOptimizerStep":
                    if not all([i.name == "InstructionOptimizerStep" for i in next_instructions_for_ranks.values()]):  # type: ignore[union-attr]
                        continue
                elif next_instruction.name in [
                    "InstructionLoadMicroBatch",
                    "InstructionForwardPass",
                    "InstructionBackwardPass",
                    "InstructionLoss",
                ]:
                    pass  # independent instructions
                else:
                    raise NotImplementedError

                next_instruction.time_start = time_spent_pipeline_stage[pipe_parallel_rank]
                next_instruction.time_end = next_instruction.time_start + next_instruction.time_duration
                time_spent_pipeline_stage[pipe_parallel_rank] = next_instruction.time_end
                next_instruction.status = SimulationStatus.DONE

            # update next instructions for ranks for next iteration
            next_instructions_for_ranks = {
                pipe_parallel_rank: self.next_instruction_for_rank(pipe_parallel_rank=pipe_parallel_rank)
                for pipe_parallel_rank in self.simulation_instructions_by_pipe_parallel_rank.keys()
            }

    def summarize(self) -> dict[str, Any]:
        results: dict[str, Any] = dict()
        results["pipe_stages"] = dict()

        # record idling time
        for (
            pipe_parallel_rank,
            simulation_instructions,
        ) in self.simulation_instructions_by_pipe_parallel_rank.items():
            idling_time = 0.0
            for i, simulation_instruction in enumerate(simulation_instructions):
                if i == 0:
                    idling_time += simulation_instruction.time_start
                else:
                    idling_time += simulation_instruction.time_start - simulation_instructions[i - 1].time_end

            results["pipe_stages"][str(pipe_parallel_rank)] = {
                "idling_time": idling_time,
                "simulated_time": simulation_instructions[-1].time_end,
                "idling_pct": idling_time / simulation_instructions[-1].time_end,
            }

        results["total_simulated_time"] = max([i["simulated_time"] for i in results["pipe_stages"].values()])
        results["total_idling_time"] = sum([i["idling_time"] for i in results["pipe_stages"].values()]) / len(
            results["pipe_stages"].keys()
        )
        results["total_idling_pct"] = results["total_idling_time"] / results["total_simulated_time"]
        return results

    def visualize(self, milliseconds_per_pixel: float = 10, pipe_pixels: int = 200) -> Image.Image:
        # get timings boundaries for resolution
        duration_min = None
        duration_max = None
        time_end_max = None
        for (
            pipe_parallel_rank,
            simulation_instructions,
        ) in self.simulation_instructions_by_pipe_parallel_rank.items():
            for simulation_instruction in simulation_instructions:
                if duration_min is None:
                    duration_min = simulation_instruction.time_duration
                else:
                    duration_min = min(duration_min, simulation_instruction.time_duration)
                if duration_max is None:
                    duration_max = simulation_instruction.time_duration
                else:
                    duration_max = max(duration_max, simulation_instruction.time_duration)
                if time_end_max is None:
                    time_end_max = simulation_instruction.time_end
                else:
                    time_end_max = max(time_end_max, simulation_instruction.time_end)

        # visualize
        assert time_end_max is not None
        image = Image.new(
            mode="RGB",
            size=(
                self.pipe_parallel_size * pipe_pixels,
                int((time_end_max // milliseconds_per_pixel) + 1),
            ),
            color=(255, 255, 255),
        )
        image_draw = ImageDraw.Draw(image)
        image_font = ImageFont.load_default()

        for (
            pipe_parallel_rank,
            simulation_instructions,
        ) in self.simulation_instructions_by_pipe_parallel_rank.items():
            for simulation_instruction in simulation_instructions:
                # derive color
                if simulation_instruction.name in [
                    "InstructionForwardPass",
                    "InstructionBackwardPass",
                ]:
                    color = (34, 139, 34)
                elif simulation_instruction.name in [
                    "InstructionSendActivation",
                    "InstructionRecvActivation",
                    "InstructionSendGrad",
                    "InstructionRecvGrad",
                ]:
                    color = (255, 255, 0)
                elif simulation_instruction.name in ["InstructionLoadMicroBatch"]:
                    color = (0, 0, 255)
                else:
                    color = (191, 191, 191)

                image_draw.rectangle(
                    (
                        (
                            pipe_parallel_rank * pipe_pixels,
                            int(simulation_instruction.time_start / milliseconds_per_pixel),
                        ),
                        (
                            (pipe_parallel_rank + 1) * pipe_pixels,
                            int(simulation_instruction.time_end / milliseconds_per_pixel),
                        ),
                    ),
                    fill=color,
                    outline=(0, 0, 0),
                    width=1,
                )
                image_draw.text(
                    (
                        (pipe_parallel_rank * pipe_pixels + 2),
                        (int(simulation_instruction.time_start / milliseconds_per_pixel) + 2),
                    ),
                    simulation_instruction.name,
                    font=image_font,
                    fill=(0, 0, 0),
                )
                image_draw.text(
                    (
                        (pipe_parallel_rank * pipe_pixels + 2),
                        (int(simulation_instruction.time_start / milliseconds_per_pixel) + 12),
                    ),
                    f"{round(simulation_instruction.time_duration, 4)}ms; mb: {simulation_instruction.micro_batch_id}",
                    font=image_font,
                    fill=(0, 0, 0),
                )

        return image

    def next_instruction_for_rank(self, pipe_parallel_rank: int) -> Optional[SimulationInstruction]:
        for simulation_instruction in self.simulation_instructions_by_pipe_parallel_rank[pipe_parallel_rank]:
            if simulation_instruction.status != SimulationStatus.DONE:
                return simulation_instruction

        return None
