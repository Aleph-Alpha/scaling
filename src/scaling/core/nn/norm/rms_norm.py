from typing import Any, Optional

import torch

from scaling.core.nn.linear.utils import gather_from_sequence_parallel_region
from scaling.core.nn.norm.layernorm import (
    LayerNormConfig,
)
from scaling.core.nn.parameter_meta import (
    UMUP_WEIGHT_TYPE,
    CoreParameterMeta,
    UMuPParameterMeta,
)
from scaling.core.nn.scale import scale_bwd, scale_fwd
from scaling.core.nn.umup import UMuParametrization
from scaling.core.topology.topology import Topology


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dimensions: int,
        device: torch.device,
        config: LayerNormConfig,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        umup_on_residual: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.eps = config.layernorm_epsilon
        self.topology = topology
        self.config = config

        self.weight = torch.nn.Parameter(torch.ones(dimensions, dtype=dtype).to(device))
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=False,
        )

        # umup parameters
        self.umup_on_residual = umup_on_residual
        self.forward_multiplier: float
        self.backward_multiplier: float
        self.weight_grad_multiplier: float
        self._use_umup: bool = False

    def _standard_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out.float().type_as(x)
        return out * weight

    def umup_setup(self, effective_batch_size: int, depth: int, **kwargs: Any) -> None:
        self._use_umup = True
        assert isinstance(
            self.umup_on_residual, bool
        ), "when using u-mup, you need to specify if norm is on a residual connection."

        if self.topology is not None:
            model_parallel_size = self.topology.config.model_parallel_size
        else:
            model_parallel_size = 1

        assert hasattr(self.weight, "core_parameter_meta")
        assert isinstance(self.weight.core_parameter_meta, CoreParameterMeta)

        self.weight.core_parameter_meta.umup_meta = UMuPParameterMeta(
            weight_type=UMUP_WEIGHT_TYPE.NORM, on_residual=self.umup_on_residual
        )

        UMuParametrization.apply_umup_to_weight(
            self.weight,
            model_parallel_size=model_parallel_size,
            effective_batch_size=effective_batch_size,
            depth=depth,
        )

        self.forward_multiplier = self.weight.core_parameter_meta.umup_meta.forward_multiplier
        self.backward_multiplier = self.weight.core_parameter_meta.umup_meta.backward_multiplier
        self.weight_grad_multiplier = self.weight.core_parameter_meta.umup_meta.grad_multiplier

    def _umup_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # pre-scaling
        x = scale_bwd(x, self.backward_multiplier)
        weight = scale_bwd(weight, self.weight_grad_multiplier)

        # linear forward
        output = self._standard_forward(x, weight)

        # post-scaling
        output = scale_fwd(output, self.forward_multiplier)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_umup:
            output = self._umup_forward(x, self.weight)
        else:
            output = self._standard_forward(x, self.weight)

        # Gather sequence parallel dim and scatter by model parallel regions
        if self.topology is not None and self.topology.config.sequence_parallel:
            output = gather_from_sequence_parallel_region(
                output, tensor_parallel_output_grad=True, topology=self.topology
            )

        return output
