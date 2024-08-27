from .base import PipelineScheduleBase
from .instructions import (
    InstructionBase,
    InstructionForwardPass,
    InstructionLoadMicroBatch,
    InstructionLoss,
    InstructionRecvActivation,
    InstructionSendActivation,
)


def _is_even(x: float) -> bool:
    return x % 2 == 0


class PipelineScheduleInference(PipelineScheduleBase):
    def instructions(self) -> list[InstructionBase]:
        instructions = []
        total_steps = self.topology.config.gradient_accumulation_steps + self.topology.config.pipe_parallel_size - 1
        for step_id in range(total_steps):
            micro_batch_id = step_id - self.topology.pipe_parallel_rank
            recv_buf, send_buf = self._build_buffers(step_id)
            instructions.extend(self._build_instructions(micro_batch_id, recv_buf, send_buf))

        return instructions

    def _build_buffers(self, step_id: int) -> tuple[int, int]:
        # Alternate send/recv buffers
        if _is_even(self.topology.pipe_parallel_rank):
            recv_buf = step_id % 2
            send_buf = (step_id + 1) % 2
        else:
            recv_buf = (step_id + 1) % 2
            send_buf = step_id % 2
        return recv_buf, send_buf

    def _build_instructions(self, micro_batch_id: int, recv_buf: int, send_buf: int) -> list[InstructionBase]:
        instructions: list[InstructionBase] = []
        is_micro_batch_valid = self._valid_micro_batch(micro_batch_id)
        is_next_micro_batch_valid = self._valid_micro_batch(micro_batch_id - 1)
        is_previous_rank_valid = self._is_valid_pipe_parallel_rank(self.topology.previous_pipe_parallel_rank)
        is_next_rank_valid = self._is_valid_pipe_parallel_rank(self.topology.next_pipe_parallel_rank)
        if is_micro_batch_valid and (
            self.topology.is_first_pipe_parallel_rank or self.topology.is_last_pipe_parallel_rank
        ):
            instructions.append(InstructionLoadMicroBatch(buffer_id=recv_buf, micro_batch_id=micro_batch_id))

        if _is_even(self.topology.pipe_parallel_rank):
            if is_next_rank_valid and is_next_micro_batch_valid:
                instructions.append(InstructionSendActivation(buffer_id=send_buf, micro_batch_id=micro_batch_id))
            if is_previous_rank_valid and is_micro_batch_valid:
                instructions.append(InstructionRecvActivation(buffer_id=recv_buf, micro_batch_id=micro_batch_id))
        else:
            if is_previous_rank_valid and is_micro_batch_valid:
                instructions.append(InstructionRecvActivation(buffer_id=recv_buf, micro_batch_id=micro_batch_id))
            if is_next_rank_valid and is_next_micro_batch_valid:
                instructions.append(InstructionSendActivation(buffer_id=send_buf, micro_batch_id=micro_batch_id))

        if is_micro_batch_valid:
            instructions.append(InstructionForwardPass(buffer_id=recv_buf, micro_batch_id=micro_batch_id))

        if self.topology.is_last_pipe_parallel_rank and is_micro_batch_valid:
            instructions.append(InstructionLoss(buffer_id=recv_buf, micro_batch_id=micro_batch_id))
        return instructions

    def required_buffer_count(self) -> int:
        """
        As many buffers as the distance from this stage to the last stage.
        """
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2
