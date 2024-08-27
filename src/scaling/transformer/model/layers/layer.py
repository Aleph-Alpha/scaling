from functools import partial
from typing import Callable, Union

import torch

from scaling.core import (
    ParallelMLP,
    ParallelSelfAttention,
    ParallelSwiGLUMLP,
    RotaryConfig,
    Topology,
    get_norm,
)

from ...context.config import (
    MLPType,
    TransformerArchitectureConfig,
)
from .base import TransformerLayerBaseIO, TransformerLayerIO


class ZeroLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def get_batched_adapter_forward(
    adapter_list: list[torch.nn.Module | None],
) -> Callable[[torch.Tensor], torch.Tensor]:
    module_list = [adapter if adapter is not None else ZeroLayer() for adapter in adapter_list]

    def forward(x: torch.Tensor) -> torch.Tensor:
        input_list = [x[i, :, :] for i in range(x.shape[0])]  # batch dimension is 0, not 1
        assert len(input_list) == len(module_list)
        output_list = [module(input) for input, module in zip(input_list, module_list)]
        return torch.stack(output_list, dim=0)

    return forward


class TransformerLayer(TransformerLayerBaseIO):
    def __init__(
        self,
        architecture_config: TransformerArchitectureConfig,
        layer_index: int,
        topology: Topology | None = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
    ) -> None:
        super().__init__()
        self.architecture_config = architecture_config
        self.topology = topology
        self.layer_index = layer_index
        bitfit_bias_name = getattr(architecture_config.bitfit_bias_config, "name", None)

        # attention components
        self.input_layernorm = get_norm(
            norm_type=architecture_config.norm_type,
            layernorm_config=self.architecture_config.layernorm,
            dimensions=self.architecture_config.hidden_size,
            device=torch.device("cuda") if topology is None else topology.device,
            dtype=architecture_config.precision.dtype,
            bitfit_bias_name=bitfit_bias_name,
            topology=self.topology,
        )

        self.self_attention = ParallelSelfAttention(
            hidden_size=self.architecture_config.hidden_size,
            num_attention_heads=self.architecture_config.num_attention_heads,
            num_local_attention_heads=self.architecture_config.num_local_attention_heads,
            local_attention_window_size=self.architecture_config.local_attention_window_size,
            masked_softmax_config=self.architecture_config.masked_softmax,
            causal=self.architecture_config.causal,
            dropout_attention_probs=self.architecture_config.dropout_attention_probs,
            rotary_config=RotaryConfig(
                dimensions=int(
                    self.architecture_config.rotary_percentage
                    * (self.architecture_config.hidden_size // self.architecture_config.num_attention_heads)
                ),
                max_seq_length=self.architecture_config.sequence_length,
                base=self.architecture_config.rotary_embedding_base,
            ),
            relative_position_embedding_type=architecture_config.relative_position_embedding_type,
            bias=self.architecture_config.attention_bias,
            topology=topology,
            dtype=self.architecture_config.precision.dtype,
            bitfit_bias_name=bitfit_bias_name,
            init_method=init_method,
            lora_config=self.architecture_config.lora_config,
            norm_type=architecture_config.norm_type,
            key_query_norm=self.architecture_config.key_query_norm,
            layernorm_config=self.architecture_config.layernorm,
            qkv_in_one=self.architecture_config.attention_qkv_in_one,
            num_kv_heads=self.architecture_config.attention_num_kv_heads,
            use_matmul=self.architecture_config.attention_use_matmul,
        )
        self.dropout_attention = torch.nn.Dropout(self.architecture_config.dropout_after_attention)

        # post-attention layernorm
        self.post_attention_layernorm = get_norm(
            norm_type=architecture_config.norm_type,
            layernorm_config=self.architecture_config.layernorm,
            dimensions=self.architecture_config.hidden_size,
            device=torch.device("cuda") if topology is None else topology.device,
            dtype=architecture_config.precision.dtype,
            bitfit_bias_name=bitfit_bias_name,
            topology=self.topology,
        )

        # mlp components
        self.mlp: Union[ParallelMLP, ParallelSwiGLUMLP]
        if architecture_config.mlp_type == MLPType.DEFAULT:
            self.mlp = ParallelMLP(
                io_features=self.architecture_config.hidden_size,
                intermediate_feature_factor=architecture_config.mlp_factor,
                bias=self.architecture_config.mlp_bias,
                topology=topology,
                dtype=self.architecture_config.precision.dtype,
                bitfit_bias_name=bitfit_bias_name,
                init_method=init_method,
            )
        elif architecture_config.mlp_type == MLPType.SWIGLU:
            self.mlp = ParallelSwiGLUMLP(
                io_features=self.architecture_config.hidden_size,
                intermediate_feature_factor=architecture_config.mlp_factor,
                bias=self.architecture_config.mlp_bias,
                topology=topology,
                dtype=self.architecture_config.precision.dtype,
                bitfit_bias_name=bitfit_bias_name,
                init_method=init_method,
            )
        else:
            raise NotImplementedError

        self.dropout_mlp = torch.nn.Dropout(self.architecture_config.dropout_after_mlp)

        # adapters
        if self.architecture_config.adapter_config is not None:
            self.load_adapter()

    def load_adapter(self) -> None:
        adapter_config = self.architecture_config.adapter_config
        assert adapter_config is not None
        if adapter_config.attention_downsampling_factor is not None:
            self.attn_adapter_name = f"attn_adapter_{adapter_config.name}"
            setattr(
                self,
                self.attn_adapter_name,
                ParallelMLP(
                    io_features=self.architecture_config.hidden_size,
                    intermediate_feature_factor=adapter_config.attention_downsampling_factor,
                    bias=False,
                    topology=self.topology,
                    dtype=self.architecture_config.precision.dtype,
                    init_method=partial(
                        torch.nn.init.normal_,
                        mean=0.0,
                        std=adapter_config.init_std,
                    ),
                ),
            )

        if adapter_config.mlp_downsampling_factor is not None:
            self.mlp_adapter_name = f"mlp_adapter_{adapter_config.name}"
            setattr(
                self,
                self.mlp_adapter_name,
                ParallelMLP(
                    io_features=self.architecture_config.hidden_size,
                    intermediate_feature_factor=adapter_config.mlp_downsampling_factor,
                    bias=False,
                    topology=self.topology,
                    dtype=self.architecture_config.precision.dtype,
                    init_method=partial(
                        torch.nn.init.normal_,
                        mean=0.0,
                        std=adapter_config.init_std,
                    ),
                ),
            )

    def apply_adapter(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        assert hasattr(self, adapter_name), f"cannot use adapter '{adapter_name}' as it is not initialized"
        adapter = getattr(self, adapter_name)
        return adapter(x)

    def attention_block(
        self,
        hidden_state: torch.Tensor,
        cumulative_seq_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = False,
        reset_cache: bool = False,
        cache_index: int = 0,
        attention_scores_manipulation: torch.Tensor | None = None,
        attentions_score_manipulation_log_additive: Union[bool, list[bool]] = True,
    ) -> torch.Tensor:
        hidden_state_tmp = self.input_layernorm(hidden_state)
        hidden_state_tmp = self.self_attention(
            hidden_state_tmp,
            cumulative_seq_lengths=cumulative_seq_lengths,
            attention_scores_manipulation=attention_scores_manipulation,
            attentions_score_manipulation_log_additive=attentions_score_manipulation_log_additive,
            position_ids=position_ids,
            use_cache=use_cache,
            reset_cache=reset_cache,
            cache_index=cache_index,
        )
        if self.topology is not None:
            with self.topology.model_parallel_constant_rng():
                hidden_state_tmp = self.dropout_attention(hidden_state_tmp)
        else:
            hidden_state_tmp = self.dropout_attention(hidden_state_tmp)
        hidden_state = hidden_state + hidden_state_tmp

        if hasattr(self, "attn_adapter_name"):
            hidden_state = hidden_state + self.apply_adapter(hidden_state, self.attn_adapter_name)

        return hidden_state

    def mlp_block(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden_state_tmp = self.post_attention_layernorm(hidden_state)
        hidden_state_tmp = self.mlp(x=hidden_state_tmp)
        if self.topology is not None:
            with self.topology.model_parallel_constant_rng():
                hidden_state_tmp = self.dropout_mlp(hidden_state_tmp)
        else:
            hidden_state_tmp = self.dropout_mlp(hidden_state_tmp)
        hidden_state = hidden_state + hidden_state_tmp

        if hasattr(self, "mlp_adapter_name"):
            hidden_state = hidden_state + self.apply_adapter(hidden_state, self.mlp_adapter_name)

        return hidden_state

    def forward(
        self,
        x: TransformerLayerIO,
    ) -> TransformerLayerIO:
        control_log_additive_batch: Union[bool, list[bool]]
        if x.inference_settings is None:
            use_cache = False
            reset_cache = False
            cache_index = 0
            control_log_additive_batch = True
        else:
            use_cache = x.inference_settings.use_cache
            reset_cache = x.inference_settings.reset_cache
            cache_index = x.inference_settings.cache_index
            control_log_additive_batch = x.inference_settings.control_log_additive_batch

        assert x.cumulative_seq_lengths is not None
        activations = self.attention_block(
            hidden_state=x.activations,
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            attention_scores_manipulation=x.attention_scores_manipulation,
            attentions_score_manipulation_log_additive=control_log_additive_batch,
            position_ids=x.position_ids,
            use_cache=use_cache,
            reset_cache=reset_cache,
            cache_index=cache_index,
        )
        activations = self.mlp_block(hidden_state=activations)

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
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,
            attention_scores_manipulation=x.attention_scores_manipulation,
            loss_weights=x.loss_weights,
            position_ids=x.position_ids,
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )
