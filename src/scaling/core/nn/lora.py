from functools import partial
from typing import Optional, Union

import torch

from scaling.core.nn.linear import ColumnParallelLinear, RowParallelLinear
from scaling.core.nn.linear.utils import all_concat
from scaling.core.nn.lora_config import LoRAModuleType
from scaling.core.topology import Topology


class ParallelLoRa(torch.nn.Module):
    """
    MLP will take the input with h hidden state, project it to the hidden dimension == rank (A)
    and project back tos h (B). With A being initialized gaussian and B zero initialized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_module_type: LoRAModuleType,
        rank: int,
        bias: bool = False,
        alpha: int = 1,
        dtype: torch.dtype = torch.float32,
        kaiming_a: float = 1.0e-5,
        device: Optional[torch.device] = None,
        dropout: Optional[float] = None,
        topology: Optional[Topology] = None,
    ) -> None:
        super().__init__()
        assert rank <= in_features, f"LoRa Rank: {rank} is greater or equal to the Input dimensionality: {in_features}"

        self.scaling = alpha / rank

        self.topology = topology

        if self.topology and self.topology.config.model_parallel_size > 1:
            assert (
                not rank % self.topology.config.model_parallel_size
            ), "If using model parallelism, make sure your rank is divisible by two."
            assert (
                not out_features % self.topology.config.model_parallel_size
            ), "If using model parallelism, make sure your out_features are divisible by two."

        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)

        self.lora_module_type = lora_module_type
        self.dense_out: Union[ColumnParallelLinear, RowParallelLinear]
        if self.topology is not None:
            self.model_parallel_size = self.topology.config.model_parallel_size
        else:
            self.model_parallel_size = 1

        if not self.lora_module_type == LoRAModuleType.DENSE:
            self.dense_in = ColumnParallelLinear(
                in_features=in_features,
                out_features=rank,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=partial(torch.nn.init.kaiming_uniform_, a=kaiming_a),
                parallel_output=False,
            )

            self.dense_out = ColumnParallelLinear(
                in_features=rank,
                out_features=out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=torch.nn.init.zeros_,
                parallel_output=True,
            )
        else:
            self.dense_in = ColumnParallelLinear(
                in_features=in_features,
                out_features=rank,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=partial(torch.nn.init.kaiming_uniform_, a=kaiming_a),
                parallel_output=True,
            )

            self.dense_out = RowParallelLinear(
                in_features=rank,
                out_features=out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=torch.nn.init.zeros_,
                parallel_input=True,
                parallel_output=False,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dense_in(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.dense_out(x)
        return x * self.scaling

    def get_delta_weights(self) -> torch.Tensor:
        use_model_parallel = self.model_parallel_size > 1
        out_dim_concat = (
            -2 if not self.lora_module_type == LoRAModuleType.DENSE else -1
        )  # needs to be done for row parallel dense out

        # Fetch dense_in and dense_out weights, adjusting for model parallelism if necessary
        dense_in = self._get_adjusted_weight(self.dense_in.weight, dim=-2, use_model_parallel=use_model_parallel)
        dense_out = self._get_adjusted_weight(
            self.dense_out.weight,
            dim=out_dim_concat,
            use_model_parallel=use_model_parallel,
        )

        # Calculate weight update
        weight_update = (dense_out @ dense_in) * self.scaling

        # Adjust weight_update for model parallelism
        if use_model_parallel:
            weight_update = self._slice_weight_update(weight_update)

        return weight_update

    def _get_adjusted_weight(
        self, weight: torch.Tensor, dim: int = -2, use_model_parallel: bool = False
    ) -> torch.Tensor:
        """
        Retrieves the adjusted weight tensor, taking model parallelism into account.
        """
        if use_model_parallel:
            assert self.topology
            return all_concat(weight, dim=dim, topology=self.topology)
        return weight

    def _slice_weight_update(self, weight_update: torch.Tensor) -> torch.Tensor:
        """
        Slices the weight_update tensor according to model parallel rank and the nature of the layer (dense or not).
        """
        if self.topology:
            i = self.topology.model_parallel_rank
        else:
            i = 0
        m, n = weight_update.size()
        n_slices = self.model_parallel_size

        if not self.lora_module_type == LoRAModuleType.DENSE:
            # For k, v, q layers: slice columns
            slice_size = m // n_slices
            return weight_update[i * slice_size : (i + 1) * slice_size]
        else:
            # For dense layers: slice rows
            slice_size = n // n_slices
            return weight_update[:, i * slice_size : (i + 1) * slice_size]
