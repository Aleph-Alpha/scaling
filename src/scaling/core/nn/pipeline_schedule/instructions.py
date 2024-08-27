from typing import NamedTuple, Optional


# Base
class InstructionBase(NamedTuple):
    buffer_id: Optional[int] = None
    micro_batch_id: Optional[int] = None
    is_first_pass: Optional[bool] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__


# Optimizer
class InstructionOptimizerStep(InstructionBase):
    pass


class InstructionReduceTiedGrads(InstructionBase):
    pass


# IO
class InstructionStoreMicroBatch(InstructionBase):
    pass


class InstructionLoadMicroBatch(InstructionBase):
    pass


# Compute
class InstructionForwardPass(InstructionBase):
    pass


# Compute
class InstructionLoss(InstructionBase):
    pass


class InstructionBackwardPass(InstructionBase):
    pass


# Communication
class InstructionSendActivation(InstructionBase):
    pass


class InstructionRecvActivation(InstructionBase):
    pass


class InstructionSendGrad(InstructionBase):
    pass


class InstructionRecvGrad(InstructionBase):
    pass
