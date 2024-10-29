# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Optional

import torch

from scaling.core.fp8 import FP8LinearConfig
from scaling.core.nn.activation_function import (
    ActivationFunction,
    get_activation_function,
)
from scaling.core.nn.linear import ColumnParallelLinear, RowParallelLinear
from scaling.core.nn.linear.utils import all_reduce_scatter_to_sequence_parallel
from scaling.core.nn.parameter_meta import UMUP_WEIGHT_TYPE
from scaling.core.nn.scale import scale_bwd, scale_fwd
from scaling.core.nn.umup import UMuParametrization
from scaling.core.topology import Topology


class ParallelMLP(torch.nn.Module):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        io_features: int,
        intermediate_feature_factor: float,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        bitfit_bias_name: Optional[str] = None,
        activation_function: ActivationFunction = ActivationFunction.GELU,
        umup_mult: Optional[float] = None,
        umup_is_first_layer: Optional[bool] = None,
        umup_is_last_layer: Optional[bool] = None,
        umup_on_residual: Optional[bool] = None,
        fp8_config: Optional[FP8LinearConfig] = None,
        fp8_config_dense_out: Optional[FP8LinearConfig] = None,
    ) -> None:
        super().__init__()

        assert (
            float(int(io_features * intermediate_feature_factor)) == io_features * intermediate_feature_factor
        ), "io_features * intermediate_feature_factor does not result in a natural number for feature dimensions"

        intermediate_features = int(io_features * intermediate_feature_factor)

        dense_in_weight_type: UMUP_WEIGHT_TYPE | None = None
        dense_out_weight_type: UMUP_WEIGHT_TYPE | None = None

        if isinstance(umup_is_first_layer, bool):
            dense_in_weight_type = (
                UMUP_WEIGHT_TYPE.INPUT_WEIGHT if umup_is_first_layer else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
            )
        if isinstance(umup_is_last_layer, bool):
            dense_out_weight_type = (
                UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT if umup_is_last_layer else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
            )

        self.dense_in = ColumnParallelLinear(
            in_features=io_features,
            out_features=intermediate_features,
            bias=bias,
            device=device,
            dtype=dtype,
            topology=topology,
            init_method=init_method,
            parallel_output=True,
            bitfit_bias_name=bitfit_bias_name,
            umup_weight_type=dense_in_weight_type,
            umup_on_residual=umup_on_residual,
            fp8_config=fp8_config,
        )

        self.dense_out = RowParallelLinear(
            in_features=intermediate_features,
            out_features=io_features,
            bias=bias,
            device=device,
            dtype=dtype,
            topology=topology,
            init_method=init_method,
            parallel_input=True,
            parallel_output=(topology.config.sequence_parallel if topology is not None else False),
            bitfit_bias_name=bitfit_bias_name,
            umup_weight_type=dense_out_weight_type,
            umup_on_residual=umup_on_residual,
            fp8_config=fp8_config_dense_out,
        )

        self.activation_function_type = activation_function
        self.activation_function = get_activation_function(self.activation_function_type)
        self.topology = topology

        self.umup_mult = umup_mult
        self._use_umup = False
        self.output_scale_factor: float

    def umup_setup(self, **kwargs: Any) -> None:
        assert self.umup_mult is not None
        self._use_umup = True
        if self.activation_function_type == ActivationFunction.GELU:
            self.output_scale_factor = UMuParametrization.get_umup_gelu_scales(self.umup_mult)
        else:
            raise NotImplementedError(
                f"UMuP currently not implemented for activation function {self.activation_function_type}."
            )

    def _umup_gelu_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = scale_bwd(x, self.output_scale_factor)
        # we divide by mult here because gelu(x) = x * Phi(x) and only Phi
        # should get the pre-multiplier
        if self.umup_mult == 1.0:
            out = torch.nn.functional.gelu(x)
        else:
            out = torch.nn.functional.gelu(self.umup_mult * x) / self.umup_mult
        out = scale_fwd(out, self.output_scale_factor)
        return out

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dense_in(x)
        if self._use_umup:
            x = self._umup_gelu_forward(x)
        else:
            x = self.activation_function(x)
        x = self.dense_out(x)

        # scatter to sequence parallel
        if self.topology is not None and self.topology.config.sequence_parallel:
            x = all_reduce_scatter_to_sequence_parallel(x, self.topology)

        return x


class ParallelSwiGLUMLP(torch.nn.Module):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        io_features: int,
        intermediate_feature_factor: float,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        bitfit_bias_name: Optional[str] = None,
        umup_mult: Optional[float] = None,
        umup_is_first_layer: Optional[bool] = None,
        umup_is_last_layer: Optional[bool] = None,
        umup_on_residual: Optional[bool] = None,
        fp8_config: Optional[FP8LinearConfig] = None,
        fp8_config_dense_out: Optional[FP8LinearConfig] = None,
    ) -> None:
        super().__init__()
        self.topology = topology

        assert (
            float(int(io_features * intermediate_feature_factor)) == io_features * intermediate_feature_factor
        ), "io_features * intermediate_feature_factor does not result in a natural number for feature dimensions"

        intermediate_features = int(io_features * intermediate_feature_factor)

        dense_in_weight_type: UMUP_WEIGHT_TYPE | None = None
        siglu_weight_type: UMUP_WEIGHT_TYPE | None = None
        dense_out_weight_type: UMUP_WEIGHT_TYPE | None = None

        if isinstance(umup_is_first_layer, bool):
            dense_in_weight_type = (
                UMUP_WEIGHT_TYPE.INPUT_WEIGHT if umup_is_first_layer else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
            )
            siglu_weight_type = UMUP_WEIGHT_TYPE.INPUT_WEIGHT if umup_is_first_layer else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
        if isinstance(umup_is_last_layer, bool):
            dense_out_weight_type = (
                UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT if umup_is_last_layer else UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT
            )

        self.dense_in = ColumnParallelLinear(
            in_features=io_features,
            out_features=intermediate_features,
            bias=bias,
            device=device,
            dtype=dtype,
            topology=topology,
            init_method=init_method,
            parallel_output=True,
            bitfit_bias_name=bitfit_bias_name,
            umup_weight_type=dense_in_weight_type,
            umup_on_residual=umup_on_residual,
            fp8_config=fp8_config,
        )

        self.siglu_weight = ColumnParallelLinear(
            in_features=io_features,
            out_features=intermediate_features,
            bias=bias,
            device=device,
            dtype=dtype,
            topology=topology,
            init_method=init_method,
            parallel_output=True,
            bitfit_bias_name=bitfit_bias_name,
            umup_weight_type=siglu_weight_type,
            umup_on_residual=umup_on_residual,
            fp8_config=fp8_config,
        )

        self.dense_out = RowParallelLinear(
            in_features=intermediate_features,
            out_features=io_features,
            bias=bias,
            device=device,
            dtype=dtype,
            topology=topology,
            init_method=init_method,
            parallel_input=True,
            parallel_output=(topology.config.sequence_parallel if topology is not None else False),
            bitfit_bias_name=bitfit_bias_name,
            umup_weight_type=dense_out_weight_type,
            umup_on_residual=umup_on_residual,
            fp8_config=fp8_config_dense_out,
        )

        # umup parameters
        self.umup_mult = umup_mult
        self._use_umup = False
        self.output_scale_factor: float

    def _standard_siglu_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x) * gate

    def umup_setup(self, **kwargs: Any) -> None:
        assert self.umup_mult is not None
        self._use_umup = True

        self.output_scale_factor = UMuParametrization.get_umup_swiglu_scales(self.umup_mult)

    def _umup_siglu_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = scale_bwd(x, self.output_scale_factor)
        gate = scale_bwd(gate, self.output_scale_factor)
        if self.umup_mult == 1.0:
            out = torch.nn.functional.silu(x) * gate
        else:
            # we divide by mult here because silu(x) = x * sigmoid(x) and only sigmoid
            # should get the pre-multiplier
            out = torch.nn.functional.silu(self.umup_mult * x) * gate / self.umup_mult
        out = scale_fwd(out, self.output_scale_factor)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_intermediate = self.dense_in(x)
        gate = self.siglu_weight(x)
        if self._use_umup:
            x_intermediate = self._umup_siglu_forward(x_intermediate, gate)
        else:
            x_intermediate = self._standard_siglu_forward(x_intermediate, gate)
        x_intermediate = self.dense_out(x_intermediate)

        # scatter to sequence parallel
        if self.topology is not None and self.topology.config.sequence_parallel:
            x_intermediate = all_reduce_scatter_to_sequence_parallel(x_intermediate, self.topology)

        return x_intermediate
