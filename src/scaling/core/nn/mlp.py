# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Optional

import torch

from scaling.core.nn.activation_function import (
    ActivationFunction,
    get_activation_function,
)
from scaling.core.nn.linear import ColumnParallelLinear, RowParallelLinear
from scaling.core.nn.linear.utils import all_reduce_scatter_to_sequence_parallel
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
    ) -> None:
        super().__init__()

        assert (
            float(int(io_features * intermediate_feature_factor)) == io_features * intermediate_feature_factor
        ), "io_features * intermediate_feature_factor does not result in a natural number for feature dimensions"

        intermediate_features = int(io_features * intermediate_feature_factor)

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
        )

        self.activation_function = get_activation_function(activation_function)
        self.topology = topology

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dense_in(x)
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
    ) -> None:
        super().__init__()
        self.topology = topology

        assert (
            float(int(io_features * intermediate_feature_factor)) == io_features * intermediate_feature_factor
        ), "io_features * intermediate_feature_factor does not result in a natural number for feature dimensions"

        intermediate_features = int(io_features * intermediate_feature_factor)

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_intermediate = self.dense_in(x)
        x_intermediate = torch.nn.functional.silu(x_intermediate)
        x_intermediate = x_intermediate * self.siglu_weight(x)
        x_intermediate = self.dense_out(x_intermediate)

        # scatter to sequence parallel
        if self.topology is not None and self.topology.config.sequence_parallel:
            x_intermediate = all_reduce_scatter_to_sequence_parallel(x_intermediate, self.topology)

        return x_intermediate
