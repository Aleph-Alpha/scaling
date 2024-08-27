import torch

from scaling.core import (
    Topology,
)

from ...context.config import TransformerArchitectureConfig
from .base import TransformerLayerBaseIO, TransformerLayerIO


class TransformerEmbeddingHead(TransformerLayerBaseIO):
    def __init__(self, architecture_config: TransformerArchitectureConfig, topology: Topology | None = None):
        super().__init__()
        assert (
            architecture_config.embedding_head_config is not None
        ), "Initializing EmbeddingHead without embedding_head and embedding_heads is not possible"
        self.architecture_config = architecture_config
        self.topology = topology

        input_size = architecture_config.hidden_size
        self.embedding_head_config = architecture_config.embedding_head_config
        for idx, layer_size in enumerate(self.embedding_head_config.proj_layers):
            setattr(
                self,
                f"embedding_head_proj_{self.embedding_head_config.name}_{idx}",
                torch.nn.Linear(
                    input_size,
                    layer_size,
                    bias=False,
                    device=torch.device("cuda") if self.topology is None else self.topology.device,
                    dtype=architecture_config.precision.dtype,
                ),
            )
            input_size = layer_size

    def forward(self, x: TransformerLayerIO) -> TransformerLayerIO:
        assert x.loss_weights is not None, "did not receive loss_weights for masking"
        last_hidden_state_pooled = self.weighted_mean_pooling(x.activations, loss_weights=x.loss_weights)
        # Pad to allow usage of differently scaling embedding heads in one batch
        if len(self.embedding_head_config.proj_layers) > 0:
            activations = self.apply_embedding_head_proj(last_hidden_state_pooled)

        return TransformerLayerIO(
            activations=activations,
            position_ids=x.position_ids,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            loss_weights=x.loss_weights,  # return the loss weights for inference so that the pooling can use atman
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )

    @staticmethod
    def weighted_mean_pooling(embeddings: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
        in_dtype = embeddings.dtype
        # Make sure dtype is fp32 due to instability
        embeddings = embeddings.to(torch.float32)
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(
                start=1,
                end=embeddings.shape[1] + 1,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(embeddings.size())
        )
        # Add hid_dim
        loss_weights_expanded = loss_weights.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        # WARNING: This is unstable in fp16, as it will lead to values > 65519 (fp16 max) hence inf
        # This could be rewritten to be more stable in fp16, but at the cost of less fraction precision
        # i.e. there will be more digits, which may be less stable in bf16 which has fewer fractions
        # Thus to be safe do it all in fp32
        sum_embeddings = torch.sum(embeddings * loss_weights_expanded * weights, dim=1)
        sum_mask = torch.sum(loss_weights_expanded * weights, dim=1)

        if sum_mask.sum() == 0.0:
            embeddings = torch.zeros_like(sum_embeddings)
        else:
            embeddings = sum_embeddings / sum_mask

        return embeddings.to(in_dtype)

    def apply_embedding_head_proj(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Apply activation after all except the last proj layer
        for idx, _ in enumerate(self.embedding_head_config.proj_layers):
            if idx > 0:
                embeddings = torch.nn.functional.gelu(embeddings)
            embeddings = getattr(self, f"embedding_head_proj_{self.embedding_head_config.name}_{idx}")(embeddings)
        return embeddings
