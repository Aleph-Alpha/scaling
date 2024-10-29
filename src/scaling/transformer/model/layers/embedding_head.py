import torch

from scaling.core import (
    Topology,
)
from scaling.transformer.context.config import PoolingMethod, TransformerArchitectureConfig

from .base import TransformerLayerBaseIO, TransformerLayerIO


class TransformerEmbeddingHead(TransformerLayerBaseIO):
    def __init__(self, architecture_config: TransformerArchitectureConfig, topology: Topology | None = None):
        super().__init__()
        assert (
            architecture_config.embedding_head_config is not None
        ), "Initializing EmbeddingHead without embedding_head and embedding_heads is not possible"
        self.embedding_head_config = architecture_config.embedding_head_config

        if self.embedding_head_config.proj_layers is not None:
            input_size = architecture_config.hidden_size
            for idx, layer_size in enumerate(self.embedding_head_config.proj_layers):
                setattr(
                    self,
                    f"embedding_head_proj_{self.embedding_head_config.name}_{idx}",
                    torch.nn.Linear(
                        input_size,
                        layer_size,
                        bias=False,
                        device=topology.device if topology else None,
                        dtype=architecture_config.precision.dtype,
                    ),
                )
                input_size = layer_size

    def forward(self, x: TransformerLayerIO) -> TransformerLayerIO:
        hidden_states = x.activations
        assert (
            x.loss_weights is not None
        ), "loss_weights of LuminousIO were None. Provide loss weights for embedding pooling"
        if self.embedding_head_config.proj_layers is not None:
            hidden_states = self.apply_embedding_head_proj(hidden_states=hidden_states)

        hidden_states = self.apply_pooling(
            hidden_states=hidden_states.to(dtype=torch.float32),
            loss_weights=x.loss_weights,
            pooling_method=self.embedding_head_config.pooling,
        )

        return TransformerLayerIO(
            activations=hidden_states,
            position_ids=x.position_ids,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            loss_weights=x.loss_weights,  # return the loss weights for inference so that the pooling can use atman
        )

    def apply_pooling(
        self, hidden_states: torch.Tensor, loss_weights: torch.Tensor, pooling_method: PoolingMethod
    ) -> torch.Tensor:
        if pooling_method == PoolingMethod.WEIGHTED_MEAN:
            return self.apply_mean_pooling(hidden_states, loss_weights=loss_weights, weighted=True)
        elif pooling_method == PoolingMethod.MEAN:
            return self.apply_mean_pooling(hidden_states, loss_weights=loss_weights)
        elif pooling_method == PoolingMethod.LAST_TOKEN:
            return self.apply_last_token_pooling(hidden_states, loss_weights=loss_weights)
        else:
            raise NotImplementedError(pooling_method)

    @staticmethod
    def apply_last_token_pooling(embeddings: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
        b, n, d = embeddings.shape
        reversed_mask = torch.flip(loss_weights, dims=(1,))
        argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
        gather_indices = loss_weights.size(1) - argmax_reverse - 1
        gather_indices = torch.clamp(gather_indices, min=0)
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
        gather_indices = gather_indices.unsqueeze(1)
        input_mask_expanded = loss_weights.unsqueeze(-1).expand((b, n, d)).float()
        embeddings = torch.gather(embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)

        return embeddings

    @staticmethod
    def apply_mean_pooling(
        embeddings: torch.Tensor, loss_weights: torch.Tensor, weighted: bool = False
    ) -> torch.Tensor:
        in_dtype = embeddings.dtype
        # Make sure dtype is fp32 due to instability
        embeddings = embeddings.to(torch.float32)
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = loss_weights.unsqueeze(-1).expand(loss_weights.size(0), loss_weights.size(1), embeddings.size(-1))

        if weighted:
            weights = weights.cumsum(dim=1)

        # Add hid_dimp
        loss_weights_expanded = loss_weights.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
        sum_embeddings = torch.sum(embeddings * loss_weights_expanded * weights, dim=1)
        sum_mask = torch.sum(loss_weights_expanded * weights, dim=1)

        if sum_mask.sum() == 0.0:
            embeddings = torch.zeros_like(sum_embeddings)
        else:
            embeddings = sum_embeddings / sum_mask

        return embeddings.to(in_dtype)

    def apply_embedding_head_proj(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.embedding_head_config.proj_layers is not None

        for idx, _ in enumerate(self.embedding_head_config.proj_layers):
            if idx > 0:
                hidden_states = torch.nn.functional.gelu(hidden_states)
            hidden_states = getattr(
                self,
                f"embedding_head_proj_{self.embedding_head_config.name}_{idx}",
            )(hidden_states)
        return hidden_states
