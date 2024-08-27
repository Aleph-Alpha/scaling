from enum import Enum
from typing import Callable

import torch


class ActivationFunction(Enum):
    GELU = "gelu"
    SILU = "silu"


def get_activation_function(activation_function: ActivationFunction) -> Callable[..., torch.Tensor]:
    if activation_function == ActivationFunction.GELU:
        return torch.nn.functional.gelu
    elif activation_function == ActivationFunction.SILU:
        return torch.nn.functional.silu
    raise NotImplementedError
