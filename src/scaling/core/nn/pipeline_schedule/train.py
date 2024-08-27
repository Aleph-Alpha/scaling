# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from HPDL group, PDL lab, NUDT
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .base import PipelineScheduleBase
from .instructions import (
    InstructionBackwardPass,
    InstructionBase,
    InstructionForwardPass,
    InstructionLoadMicroBatch,
    InstructionLoss,
    InstructionOptimizerStep,
    InstructionRecvActivation,
    InstructionRecvGrad,
    InstructionReduceTiedGrads,
    InstructionSendActivation,
    InstructionSendGrad,
)


def _is_even(x: float) -> bool:
    return x % 2 == 0


def _is_odd(x: float) -> bool:
    return x % 2 != 0


class PipelineScheduleTrain(PipelineScheduleBase):
    def instructions(self) -> list[InstructionBase]:
        instructions: list[InstructionBase] = list()
        prev_micro_batch_id = -1

        # Compute the number of total steps required
        # - self.topology.config.gradient_accumulation_steps are needed for the first pipe parallel layer
        # for input of each micro batch
        # - each micro batch must run through the pipeline twice for forward and backward pass
        total_steps = 2 * (
            self.topology.config.gradient_accumulation_steps + self.topology.config.pipe_parallel_size - 1
        )
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._is_valid_pipe_parallel_rank(
                    self.topology.previous_pipe_parallel_rank
                ):
                    instructions.append(InstructionRecvActivation(buffer_id=curr_buffer, micro_batch_id=micro_batch_id))
                if self._valid_micro_batch(prev_micro_batch_id) and self._is_valid_pipe_parallel_rank(
                    self.topology.previous_pipe_parallel_rank
                ):
                    instructions.append(InstructionSendGrad(buffer_id=prev_buffer, micro_batch_id=prev_micro_batch_id))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._is_valid_pipe_parallel_rank(
                    self.topology.next_pipe_parallel_rank
                ):
                    instructions.append(
                        InstructionSendActivation(buffer_id=prev_buffer, micro_batch_id=prev_micro_batch_id)
                    )
                if self._valid_micro_batch(micro_batch_id) and self._is_valid_pipe_parallel_rank(
                    self.topology.next_pipe_parallel_rank
                ):
                    instructions.append(InstructionRecvGrad(buffer_id=curr_buffer, micro_batch_id=micro_batch_id))

            # First/last stage loads
            if (
                self.topology.pipe_parallel_rank == 0
                or self.topology.pipe_parallel_rank == self.topology.config.pipe_parallel_size - 1
            ):
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    instructions.append(InstructionLoadMicroBatch(buffer_id=curr_buffer, micro_batch_id=micro_batch_id))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    instructions.append(InstructionForwardPass(buffer_id=curr_buffer, micro_batch_id=micro_batch_id))
                    if self.topology.is_last_pipe_parallel_rank:
                        instructions.append(
                            InstructionLoss(
                                buffer_id=curr_buffer,
                                micro_batch_id=micro_batch_id,
                                is_first_pass=True,
                            )
                        )
                else:
                    instructions.append(InstructionBackwardPass(buffer_id=curr_buffer, micro_batch_id=micro_batch_id))

            if step_id == total_steps - 1:
                # reduce gradients
                instructions.append(InstructionReduceTiedGrads())
                instructions.append(InstructionOptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id

        return instructions

    def required_buffer_count(self) -> int:
        """
        As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(
            self.topology.config.pipe_parallel_size - self.topology.pipe_parallel_rank + 1,
            self.topology.config.gradient_accumulation_steps,
        )
        return max(2, buffers)

    def _buffer_idx(self, micro_batch_id: int) -> int:
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.required_buffer_count()

    def _step_to_micro_batch(self, step_id: int) -> tuple[int, bool]:
        if _is_even(step_id) and _is_even(self.topology.pipe_parallel_rank):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.topology.pipe_parallel_rank):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.topology.pipe_parallel_rank):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.topology.pipe_parallel_rank):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False
        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id: int) -> int:
        base = step_id // 2
        micro_batch_id = int(base - self.topology.pipe_parallel_rank // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id: int) -> int:
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.topology.pipe_parallel_rank // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id: int) -> int:
        base = step_id // 2
        micro_batch_id = int(
            base - self.topology.config.pipe_parallel_size + (self.topology.pipe_parallel_rank + 1) // 2
        )
        return micro_batch_id

    def _odd_step_backward_id(self, step_id: int) -> int:
        base = ((step_id - 1) // 2) - self.topology.config.pipe_parallel_size + 1
        micro_batch_id = int(base + self.topology.pipe_parallel_rank // 2)
        return micro_batch_id
