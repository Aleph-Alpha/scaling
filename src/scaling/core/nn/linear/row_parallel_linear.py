from typing import Callable, Optional

import torch

from ...topology import Topology
from ..parameter_meta import (
    CoreParameterMeta,
)
from .utils import (
    all_reduce,
    all_shard,
    get_device,
)


class RowParallelLinear(torch.nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
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
        parallel_input: bool = False,
        parallel_output: bool = False,
        bitfit_bias_name: Optional[str] = None,
    ) -> None:
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
        parallel_input (`bool`)
            If true, input is parallel and there is no all-shard across the partitions
            at the beginning of the forward pass.
            This means that the row parallel linear can follow a column parallel linear layer
            with parallel output in the column parallel linear.
            Default: False
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
        self.parallel_input = parallel_input
        self.parallel_output = parallel_output

        # determine size for active model parallel
        self.model_parallel_size = 1 if self.topology is None else self.topology.config.model_parallel_size
        assert self.in_features % self.model_parallel_size == 0, (
            f"cannot row parallelize, in_features ({in_features}) "
            f"needs to be divisible by model parallel size ({self.model_parallel_size})"
        )
        self.input_features_per_partition = self.in_features // self.model_parallel_size

        # initialize parameters
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.out_features,
                self.input_features_per_partition,
                device=self._device,
                dtype=self.dtype,
            )
        )
        # initialize weights
        init_method(self.weight)
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=True,
            model_parallel_dimension=1,
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
                        self.out_features,
                        device=self._device,
                        dtype=self.dtype,
                    )
                ),
            )
            CoreParameterMeta.register_on_parameter(
                parameter=getattr(self, self.bias_name),
                is_model_parallel=False,
            )

        else:
            self.bias_name = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        # Set up backprop all-reduce.
        if self.model_parallel_size > 1 and not self.parallel_input:
            assert self.topology is not None
            x_parallel = all_shard(x, dim=-1, topology=self.topology)
        else:
            x_parallel = x

        if self.bias_name:
            bias = getattr(
                self,
                self.bias_name,
            )
        else:
            bias = None

        output_parallel = torch.nn.functional.linear(x_parallel, weight)

        if self.parallel_output or self.topology is None:
            output = output_parallel
        else:
            # All-reduce across all the partitions.
            output = all_reduce(output_parallel, topology=self.topology)

        if bias is not None:
            output = output + bias

        return output
