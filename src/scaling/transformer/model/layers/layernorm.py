import torch

from scaling.core import (
    Topology,
    get_norm,
)
from scaling.transformer.context.config import TransformerArchitectureConfig
from scaling.transformer.model.layers.base import TransformerLayerBaseIO, TransformerLayerIO


class LayerNormWrapper(TransformerLayerBaseIO):
    def __init__(
        self,
        architecture_config: TransformerArchitectureConfig,
        layer_index: int,
        topology: Topology | None = None,
        umup_on_residual: bool | None = None,
    ):
        super().__init__()
        self.topology = topology
        self.architecture_config = architecture_config
        bitfit_bias_name = getattr(architecture_config.bitfit_bias_config, "name", None)
        self.norm = get_norm(
            norm_type=architecture_config.norm_type,
            layernorm_config=self.architecture_config.layernorm,
            dimensions=self.architecture_config.hidden_size,
            device=torch.device("cuda") if topology is None else topology.device,
            dtype=architecture_config.precision.dtype,
            bitfit_bias_name=bitfit_bias_name,
            topology=topology,
            umup_on_residual=umup_on_residual,
        )

        self.layer_index = layer_index

    def forward(self, x: TransformerLayerIO) -> TransformerLayerIO:
        activations = self.norm(x.activations)

        if x.inference_settings is not None and (self.layer_index + 1) in x.inference_settings.embedding_layers:
            embeddings = x.embeddings
            assert embeddings is not None
            embeddings[
                x.inference_settings.embedding_layers.index(self.layer_index + 1),
                :,
                :,
                :,
            ] = activations
        else:
            embeddings = x.embeddings

        return TransformerLayerIO(
            activations=activations,
            position_ids=x.position_ids,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            loss_weights=x.loss_weights,
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )
