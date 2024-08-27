from typing import Optional

import torch

from scaling.core.nn.linear import gather_from_sequence_parallel_region
from scaling.core.topology import Topology

from .layernorm_config import LayerNormConfig, LayerNormOptimizationType

try:
    from flash_attn.ops.rms_norm import rms_norm as flash_attn_rms_norm
except ImportError:
    flash_attn_rms_norm = None
    print("Cannot import flash-attention fused kernels for RMSNorm")

from ..parameter_meta import (
    CoreParameterMeta,
)


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dimensions: int,
        device: torch.device,
        config: LayerNormConfig,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
    ) -> None:
        super().__init__()
        self.eps = config.layernorm_epsilon
        self.topology = topology
        self.config = config

        assert (
            flash_attn_rms_norm is not None or self.config.optimization_type == LayerNormOptimizationType.TORCH
        ), "The fused rms norm can only be used if the cuda kernels repo was installed."

        self.weight = torch.nn.Parameter(torch.ones(dimensions, dtype=dtype).to(device))
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=False,
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        if self.config.optimization_type == LayerNormOptimizationType.TORCH:
            output = self._norm(x.float()).type_as(x)
            output = output * weight
        else:
            output = flash_attn_rms_norm(x, self.weight, self.eps)

        # Gather sequence parallel dim and scatter by model parallel regions
        if self.topology is not None and self.topology.config.sequence_parallel:
            output = gather_from_sequence_parallel_region(
                output, tensor_parallel_output_grad=True, topology=self.topology
            )

        return output
