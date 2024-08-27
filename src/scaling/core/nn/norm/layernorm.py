from typing import Optional

import torch

from scaling.core.nn.linear import gather_from_sequence_parallel_region

from ...topology import Topology
from ..parameter_meta import (
    CoreParameterMeta,
)
from .layernorm_config import LayerNormConfig, LayerNormOptimizationType


class LayerNorm(torch.nn.Module):
    """
    TODO: docstring
    https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        config: LayerNormConfig,
        normalized_shape: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        bitfit_bias_name: Optional[str] = None,
        topology: Optional[Topology] = None,
    ):
        super().__init__()

        self.config = config
        if bitfit_bias_name is None or bitfit_bias_name == "":
            self.bias_name = "bias"
        else:
            self.bias_name = f"bias_{bitfit_bias_name}"

        self.normalized_shape = torch.Size(((normalized_shape,)))

        self.topology = topology

        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=False,
        )

        setattr(
            self,
            self.bias_name,
            torch.nn.Parameter(
                torch.zeros(
                    self.normalized_shape,
                    device=device,
                    dtype=dtype,
                )
            ),
        )
        CoreParameterMeta.register_on_parameter(
            parameter=getattr(self, self.bias_name),
            is_model_parallel=False,
        )

        if self.config.optimization_type == LayerNormOptimizationType.TORCH:
            self.norm = torch.nn.functional.layer_norm
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        bias = getattr(self, self.bias_name)

        output = self.norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            self.config.layernorm_epsilon,
        )

        # Gather sequence parallel dim and scatter by model parallel regions
        if self.topology is not None and self.topology.config.sequence_parallel:
            output = gather_from_sequence_parallel_region(
                output, tensor_parallel_output_grad=True, topology=self.topology
            )

        return output
