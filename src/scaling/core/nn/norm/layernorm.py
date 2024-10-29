from typing import Any, Optional, Sequence

import torch

from scaling.core.nn.linear import gather_from_sequence_parallel_region
from scaling.core.nn.norm.layernorm_config import LayerNormConfig, LayerNormOptimizationType
from scaling.core.nn.parameter_meta import (
    UMUP_WEIGHT_TYPE,
    CoreParameterMeta,
    UMuPParameterMeta,
)
from scaling.core.nn.scale import scale_bwd, scale_fwd
from scaling.core.nn.umup import UMuParametrization
from scaling.core.topology import Topology


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
        umup_on_residual: Optional[bool] = None,
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

        # umup parameters
        self.umup_on_residual = umup_on_residual
        self.forward_multiplier: float
        self.backward_multiplier: float
        self.weight_grad_multiplier: float
        self.bias_grad_multiplier: float
        self._use_umup: bool = False

    def _standard_forward(
        self,
        x: torch.Tensor,
        normalized_shape: Sequence[int],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        if self.config.optimization_type == LayerNormOptimizationType.TORCH:
            return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
        else:
            raise NotImplementedError

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

        if self.bias_name is not None:
            bias = getattr(
                self,
                self.bias_name,
            )

            assert hasattr(bias, "core_parameter_meta")
            assert isinstance(bias.core_parameter_meta, CoreParameterMeta)

            bias.core_parameter_meta.umup_meta = UMuPParameterMeta(
                weight_type=UMUP_WEIGHT_TYPE.BIAS, on_residual=self.umup_on_residual
            )

            UMuParametrization.apply_umup_to_weight(
                bias,
                model_parallel_size=model_parallel_size,
                effective_batch_size=effective_batch_size,
                depth=depth,
            )

            self.bias_grad_multiplier = bias.core_parameter_meta.umup_meta.grad_multiplier
        else:
            self.bias_grad_multiplier = 1.0

    def _umup_forward(
        self,
        x: torch.Tensor,
        normalized_shape: Sequence[int],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        # pre-scaling
        x = scale_bwd(x, self.backward_multiplier)
        weight = scale_bwd(weight, self.weight_grad_multiplier)
        bias = scale_bwd(bias, self.bias_grad_multiplier) if bias is not None else None

        # linear forward
        output = self._standard_forward(x, normalized_shape, weight, bias, eps)

        # post-scaling
        output = scale_fwd(output, self.forward_multiplier)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        bias = getattr(self, self.bias_name)

        if self._use_umup:
            output = self._umup_forward(
                x,
                self.normalized_shape,
                weight,
                bias,
                self.config.layernorm_epsilon,
            )
        else:
            output = self._standard_forward(
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
