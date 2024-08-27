import collections
from enum import Enum
from typing import Any, Optional

import torch


class BufferType(Enum):
    PIPELINE_STAGE_INPUT = "pipeline_stage_input"
    PIPELINE_STAGE_OUTPUT = "pipeline_stage_output"
    TARGET = "pipeline_stage_target"
    LOSS = "loss"
    METRICS = "metrics"
    GRAD = "grad"


class Buffers:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.data: dict[BufferType, dict[int, Optional[Any]]] = collections.defaultdict(dict)
        self.accum_loss: Optional[torch.Tensor] = None

    def write(self, buffer_type: BufferType, buffer_id: int, data: Any) -> None:
        self.data[buffer_type][buffer_id] = data

    def get(self, buffer_type: BufferType, buffer_id: int) -> Any:
        data = self.data[buffer_type][buffer_id]
        return data

    def take(self, buffer_type: BufferType, buffer_id: int) -> Any:
        data = self.data[buffer_type][buffer_id]
        self.data[buffer_type][buffer_id] = None

        return data

    def dump(self, buffer_type: BufferType) -> dict[int, Any]:
        data = self.data[buffer_type]
        return data

    def add_loss(self, loss: torch.Tensor) -> None:
        if self.accum_loss is None:
            self.accum_loss = loss.clone().detach()
        else:
            self.accum_loss = self.accum_loss + loss.clone().detach()
