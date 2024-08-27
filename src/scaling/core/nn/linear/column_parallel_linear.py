# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Optional

import torch

from ...topology import Topology
from ..parameter_meta import (
    CoreParameterMeta,
)
from .utils import (
    all_concat,
    copy_to_tensor_model_parallel_region,
    get_device,
)


class ColumnParallelLinear(torch.nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        parallel_output: bool = False,
        bitfit_bias_name: Optional[str] = None,
    ):
        """
        in_features (`int`)
            first dimension of matrix A.
        out_features (`int`)
            second dimension of matrix A.
        bias (`bool`)
            If true, add bias to the linear layer
        device (`Optional[torch.device]`)
            Current torch device
        dtype (`torch.dtype`)
            Data type of created weights and biases of linear layer
            Default: torch.float32
        topology (`Optional[Topology]`)
            Layout on nodes
        init_method (`Callable[[torch.Tensor], torch.Tensor]`)
            initialization method for linear layers.
            Default: torch.nn.init.xavier_normal_
        parallel_output (`bool`)
            If true, output is parallel and there is no all-gather across the partitions at the end of the forward pass
            Default: False
        """
        super().__init__()

        # remember parameters
        self.in_features = in_features
        self.out_features = out_features

        self._device = get_device(
            topology=topology,
            device=device,
        )

        self.dtype = dtype
        self.topology = topology
        self.init_method = init_method
        self.parallel_output = parallel_output

        # determine size for active model parallel
        self.model_parallel_size = 1 if self.topology is None else self.topology.config.model_parallel_size
        assert self.out_features % self.model_parallel_size == 0, (
            f"cannot column parallelize, out_features ({out_features}) "
            f"needs to be divisible by model parallel size ({self.model_parallel_size})"
        )
        self.output_features_per_partition = self.out_features // self.model_parallel_size

        # initialize parameters
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.output_features_per_partition,
                self.in_features,
                device=self._device,
                dtype=self.dtype,
            )
        )
        # initialize weights
        init_method(self.weight)
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=True,
            model_parallel_dimension=0,
        )

        self.bias_name: str | None = None
        if bias:
            # Always initialize bias to zero.
            if bitfit_bias_name is None or bitfit_bias_name == "":
                self.bias_name = "bias"
            else:
                self.bias_name = f"bias_{bitfit_bias_name}"

            setattr(
                self,
                self.bias_name,
                torch.nn.Parameter(
                    torch.zeros(
                        self.output_features_per_partition,
                        device=self._device,
                        dtype=self.dtype,
                    )
                ),
            )
            CoreParameterMeta.register_on_parameter(
                parameter=getattr(self, self.bias_name),
                is_model_parallel=True,
                model_parallel_dimension=0,
            )

        else:
            self.bias_name = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        # Set up backprop all-reduce.
        if self.model_parallel_size > 1 and self.topology is not None and not self.topology.config.sequence_parallel:
            assert self.topology is not None
            x_parallel = copy_to_tensor_model_parallel_region(x, topology=self.topology)
        else:
            x_parallel = x

        if self.bias_name is not None:
            bias = getattr(
                self,
                self.bias_name,
            )
        else:
            bias = None

        output_parallel = torch.nn.functional.linear(x_parallel, weight, bias)

        if self.parallel_output or self.topology is None:
            output = output_parallel
        else:
            # All-gather across the partitions.
            output = all_concat(output_parallel, dim=-1, topology=self.topology)

        return output
