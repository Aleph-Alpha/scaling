from typing import Callable

import torch

from scaling.core import (
    Topology,
    VocabParallelEmbedding,
)
from scaling.core.nn.linear.utils import all_concat, copy_to_tensor_model_parallel_region

from ...context.config import (
    TransformerArchitectureConfig,
)
from .base import TransformerLayerBaseIO, TransformerLayerIO


class TransformerLMHeadTied(TransformerLayerBaseIO):
    def __init__(
        self,
        architecture_config: TransformerArchitectureConfig,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        topology: Topology | None = None,
    ):
        super().__init__()
        self.topology = topology

        self.embedding = VocabParallelEmbedding(
            num_embeddings=architecture_config.vocab_size,
            embedding_dim=architecture_config.hidden_size,
            topology=self.topology,
            dtype=architecture_config.precision.dtype,
            init_method=init_method,
            finetunable_token_ids=architecture_config.finetunable_token_ids,
        )

    def forward(self, x: TransformerLayerIO) -> TransformerLayerIO:
        activations = x.activations
        if self.topology is not None and self.topology.config.model_parallel_size > 1:
            activations = copy_to_tensor_model_parallel_region(activations, topology=self.topology)
        weight = self.embedding.weight
        activations_language_tokens = torch.nn.functional.linear(activations, weight)
        if self.topology is not None and self.topology.config.model_parallel_size > 1:
            activations_language_tokens = all_concat(activations_language_tokens, dim=-1, topology=self.topology)
        activations = activations_language_tokens

        return TransformerLayerIO(
            activations=activations,
            position_ids=x.position_ids,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            loss_weights=x.loss_weights,
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )
