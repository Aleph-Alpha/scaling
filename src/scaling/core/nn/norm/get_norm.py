from enum import Enum
from typing import Optional, Union

import torch

from scaling.core.topology import Topology

from .layernorm import LayerNorm, LayerNormConfig
from .rms_norm import RMSNorm


class NormType(Enum):
    LAYERNORM = "layernorm"
    RMS = "rms"


def get_norm(
    norm_type: NormType,
    layernorm_config: Optional[LayerNormConfig],
    dimensions: int,
    device: torch.device,
    dtype: torch.dtype,
    bitfit_bias_name: Optional[str] = None,
    topology: Optional[Topology] = None,
) -> Union[LayerNorm, RMSNorm]:
    if norm_type == NormType.LAYERNORM:
        assert layernorm_config is not None
        return LayerNorm(
            config=layernorm_config,
            normalized_shape=dimensions,
            device=device,
            dtype=dtype,
            bitfit_bias_name=bitfit_bias_name,
            topology=topology,
        )
    elif norm_type == NormType.RMS:
        assert layernorm_config is not None
        return RMSNorm(
            config=layernorm_config,
            dimensions=dimensions,
            device=device,
            dtype=dtype,
            topology=topology,
        )
    else:
        raise NotImplementedError(f"{norm_type} {layernorm_config.optimization_type if layernorm_config else ''}")
