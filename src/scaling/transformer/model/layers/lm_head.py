from typing import Callable

import torch

from scaling.core import (
    ColumnParallelLinear,
    Topology,
)

from ...context.config import (
    TransformerArchitectureConfig,
)
from .base import TransformerLayerBaseIO, TransformerLayerIO


class TransformerLMHead(TransformerLayerBaseIO):
    def __init__(
        self,
        architecture_config: TransformerArchitectureConfig,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        topology: Topology | None = None,
    ):
        super().__init__()
        self.topology = topology
        self.linear = ColumnParallelLinear(
            in_features=architecture_config.hidden_size,
            out_features=architecture_config.vocab_size,
            bias=False,
            topology=topology,
            dtype=architecture_config.precision.dtype,
            init_method=init_method,
        )

        if len(architecture_config.finetunable_token_ids) > 0:
            model_parallel_size = 1 if topology is None else topology.config.model_parallel_size
            model_parallel_rank = 0 if topology is None else topology.model_parallel_rank
            vocab_size_per_partition = architecture_config.vocab_size // model_parallel_size
            mask = torch.zeros_like(self.linear.weight)
            for token_id in architecture_config.finetunable_token_ids:
                if (
                    (vocab_size_per_partition * model_parallel_rank)
                    <= token_id
                    < (vocab_size_per_partition * (model_parallel_rank + 1))
                ):
                    mask[token_id - (vocab_size_per_partition * model_parallel_rank), :] = 1

            def weight_hook(grad: torch.Tensor | None) -> torch.Tensor | None:
                if grad is None:
                    return None
                else:
                    return mask * grad

            self.linear.weight.register_hook(weight_hook)

    def forward(self, x: TransformerLayerIO) -> TransformerLayerIO:
        logits = self.linear(x.activations)
        return TransformerLayerIO(
            activations=logits,
            position_ids=x.position_ids,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            loss_weights=x.loss_weights,
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )
