# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Any, Callable, Optional

import torch

from scaling.core.fp8 import FP8LinearConfig, fp8_linear
from scaling.core.nn.linear.utils import (
    all_concat,
    copy_to_tensor_model_parallel_region,
    get_device,
)
from scaling.core.nn.parameter_meta import (
    UMUP_WEIGHT_TYPE,
    CoreParameterMeta,
    UMuPParameterMeta,
)
from scaling.core.nn.scale import scale_bwd, scale_fwd
from scaling.core.nn.umup import UMuParametrization
from scaling.core.topology import Topology


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
        umup_weight_type: Optional[UMUP_WEIGHT_TYPE] = None,
        umup_on_residual: Optional[bool] = None,
        fp8_config: Optional[FP8LinearConfig] = None,
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
        self.fp8_config = fp8_config

        # determine size for active model parallel
        self.model_parallel_size = 1 if self.topology is None else self.topology.config.model_parallel_size
        assert self.out_features % self.model_parallel_size == 0, (
            f"cannot column parallelize, out_features ({out_features}) "
            f"needs to be divisible by model parallel size ({self.model_parallel_size})"
        )
        self.output_features_per_partition = self.out_features // self.model_parallel_size

        # Initialize weights. When loading in fp8, we create fp8 weights and avoid the init method.
        self.load_in_fp8 = False
        weight_dtype = self.dtype
        if self.fp8_config is not None and self.fp8_config.dtypes_forward is not None and self.fp8_config.load_in_fp8:
            self.load_in_fp8 = True
            weight_dtype = self.fp8_config.dtypes_forward.right_dtype.torch_dtype

            def dummy_init_(x: torch.Tensor) -> torch.Tensor:
                return x

            init_method = dummy_init_
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.output_features_per_partition,
                self.in_features,
                device=self._device,
                dtype=weight_dtype,
            )
        )
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

        # umup parameters
        self.umup_weight_type = umup_weight_type
        self.umup_on_residual = umup_on_residual
        self.forward_multiplier: float
        self.backward_multiplier: float
        self.weight_grad_multiplier: float
        self.bias_grad_multiplier: float
        self._use_umup: bool = False

        if self.fp8_config is None:
            self._linear = torch.nn.functional.linear
        elif isinstance(fp8_config, FP8LinearConfig):
            self._linear = partial(
                fp8_linear,
                dtypes_forward=self.fp8_config.torch_dtypes_forward,
                dtypes_grad_input=self.fp8_config.torch_dtypes_grad_input,
                dtypes_grad_weight=self.fp8_config.torch_dtypes_grad_weight,
            )
        else:
            raise ValueError(f"Unknown option for fp8_config: {self.fp8_config}.")

    def _standard_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return self._linear(x, weight, bias)

    def umup_setup(self, effective_batch_size: int, depth: int, **kwargs: Any) -> None:
        self._use_umup = True

        assert self.umup_weight_type in (
            UMUP_WEIGHT_TYPE.INPUT_WEIGHT,
            UMUP_WEIGHT_TYPE.HIDDEN_WEIGHT,
            UMUP_WEIGHT_TYPE.OUTPUT_WEIGHT,
        ), "when using u-mup, the weight type of the linear layer needs to be provided as input, hidden or output."
        assert isinstance(
            self.umup_on_residual, bool
        ), "when using u-mup, you need to specify if linear layer is on a residual connection."

        if self.topology is not None:
            model_parallel_size = self.topology.config.model_parallel_size
        else:
            model_parallel_size = 1

        assert hasattr(self.weight, "core_parameter_meta")
        assert isinstance(self.weight.core_parameter_meta, CoreParameterMeta)

        self.weight.core_parameter_meta.umup_meta = UMuPParameterMeta(
            weight_type=self.umup_weight_type, on_residual=self.umup_on_residual
        )

        UMuParametrization.apply_umup_to_weight(
            self.weight,
            model_parallel_size=model_parallel_size,
            effective_batch_size=effective_batch_size,
            depth=depth,
            reinitialize=not (self.load_in_fp8),
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

    def _umup_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        # pre-scaling
        x = scale_bwd(x, self.backward_multiplier)
        weight = scale_bwd(weight, self.weight_grad_multiplier)
        bias = scale_bwd(bias, self.bias_grad_multiplier) if bias is not None else None

        # linear forward
        output = self._standard_forward(x, weight, None)

        # post-scaling
        output = scale_fwd(output, self.forward_multiplier)

        if bias is not None:
            output = output + bias

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        # Set up backprop all-reduce.
        if (
            self.model_parallel_size > 1
            and self.topology is not None
            and self.topology.is_distributed_initialized
            and not self.topology.config.sequence_parallel
        ):
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

        if self._use_umup:
            output_parallel = self._umup_forward(x_parallel, weight, bias)
        else:
            output_parallel = self._standard_forward(x_parallel, weight, bias)

        if self.parallel_output or self.topology is None:
            output = output_parallel
        else:
            # All-gather across the partitions.
            output = all_concat(output_parallel, dim=-1, topology=self.topology)

        return output
