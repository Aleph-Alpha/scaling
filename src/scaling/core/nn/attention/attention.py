# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import math
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from scaling.core.nn.linear import ColumnParallelLinear, RowParallelLinear
from scaling.core.nn.linear.utils import (
    all_concat,
    all_reduce_scatter_to_sequence_parallel,
    all_shard,
)
from scaling.core.nn.lora import ParallelLoRa
from scaling.core.nn.lora_config import LoRaConfig, LoRAModuleType
from scaling.core.nn.masked_softmax import MaskedSoftmax, MaskedSoftmaxConfig, MaskedSoftmaxKernel
from scaling.core.nn.norm import LayerNorm, LayerNormConfig, NormType, RMSNorm, get_norm
from scaling.core.nn.rotary import RotaryConfig, RotaryEmbedding, RotaryEmbeddingComplex
from scaling.core.topology import Topology

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except Exception as e:
    print(
        f"Could not import flash attention. "
        f"This is expected if the optional 'gpu_optimization' dependency is not installed. {e}"
    )
    flash_attn_varlen_func = None


def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int) -> Tuple[torch.Tensor, ...]:
    """
    Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    return tuple(torch.split(tensor, last_dim_size, dim=last_dim))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def get_max_seq_length(cumulative_seq_lengths: torch.Tensor) -> int:
    return int((cumulative_seq_lengths[1:] - cumulative_seq_lengths[:-1]).max().item())


def cumulative_seq_lengths_to_dense_attention_mask(
    cumulative_seq_lengths: torch.Tensor, seq_length_per_batch_item: int, causal: bool
) -> torch.Tensor:
    # Compute block sizes per batch item.
    block_sizes = cumulative_seq_lengths[1:] - cumulative_seq_lengths[:-1]
    (batch_item_cut_offs,) = torch.where(cumulative_seq_lengths % seq_length_per_batch_item == 0)
    block_sizes_per_batch_item = [
        block_sizes[start:end] for start, end in zip(batch_item_cut_offs[:-1], batch_item_cut_offs[1:])
    ]

    # Construct attention mask.
    attention_mask = torch.stack(
        [
            torch.block_diag(
                *[
                    torch.ones(block_size, block_size, dtype=torch.bool, device=cumulative_seq_lengths.device)
                    for block_size in block_sizes
                ]
            )
            for block_sizes in block_sizes_per_batch_item
        ]
    )
    if causal:
        attention_mask = torch.tril(attention_mask)
    return ~attention_mask.unsqueeze(1)


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative_seq_lengths: torch.Tensor,
    causal: bool,
    query_key_scaling_factor: float,
    softmax_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dropout_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    attention_scores_manipulation: Optional[torch.Tensor] = None,
    attentions_score_manipulation_log_additive: Union[bool, List[bool]] = True,
    use_matmul: bool = False,
) -> torch.Tensor:
    """
    Compute attention.

    Args:
        query (Tensor [s_q b n h])
        key (Tensor [s_k b n h])
        value (Tensor [s_k b n h])
    Returns:
        Tensor[s_q, b, hp] : context_layer
        Tensor[b np s_q s_k] : attention_probs
    """
    # [s_q, b, np, hn] -> [s_q, b * np, hn]
    batch_size = query.shape[1]
    seq_length = query.shape[0]
    query = rearrange(query, "s_q b n h -> s_q (b n) h")
    key = rearrange(key, "s_k b n h -> s_k (b n) h")
    value = rearrange(value, "s_k b n h -> s_k (b n) h")

    # pre allocating result tensor: [b * np, s_q, s_k]
    if use_matmul:
        matmul_result = (
            torch.matmul(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) * query_key_scaling_factor
        )
    else:
        matmul_result = torch.empty(
            query.size(1),
            query.size(0),
            key.size(0),
            dtype=query.dtype,
            device=query.device,
        )

        # Raw attention scores. [b * np, s_q, s_k]
        matmul_result = torch.baddbmm(
            matmul_result,
            query.transpose(0, 1),  # [b * np, s_q, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, s_k]
            beta=0.0,
            alpha=query_key_scaling_factor,
        )

    # change view from [b * np, s_q, s_k] to [b, np, s_q, s_k]
    attention_scores = rearrange(matmul_result, "(b n) s_q s_k -> b n s_q s_k", b=batch_size)

    # Materialize attention mask
    attention_mask = cumulative_seq_lengths_to_dense_attention_mask(
        cumulative_seq_lengths, seq_length_per_batch_item=seq_length, causal=causal
    )

    if attention_scores_manipulation is not None:
        if isinstance(attentions_score_manipulation_log_additive, bool) and attentions_score_manipulation_log_additive:
            attention_scores = attention_scores + attention_scores_manipulation
        elif (
            isinstance(attentions_score_manipulation_log_additive, bool)
            and not attentions_score_manipulation_log_additive
        ):
            # shift attention scores such that the min value is always 0
            attention_scores = attention_scores - attention_scores.masked_fill_(attention_mask, +10000.0).min(
                -1
            ).values.unsqueeze(3)
            ## apply modified mask after repeating it along seq dim i.e attention_scores.shape[1]
            attention_scores = attention_scores * attention_scores_manipulation
        else:
            assert isinstance(attentions_score_manipulation_log_additive, list)
            assert len(attentions_score_manipulation_log_additive) == attention_scores.shape[0]
            for batch_index, log_additive in enumerate(attentions_score_manipulation_log_additive):
                if log_additive:
                    attention_scores[batch_index] = (
                        attention_scores[batch_index] + attention_scores_manipulation[batch_index]
                    )
                else:
                    # shift attention scores such that the min value is always 0
                    attention_scores[batch_index] = attention_scores[batch_index] - attention_scores[
                        batch_index
                    ].masked_fill_(
                        attention_mask[batch_index],
                        +10000.0,
                    ).min(-1).values.unsqueeze(2)
                    ## apply modified mask after repeating it along seq dim i.e attention_scores.shape[1]
                    attention_scores[batch_index] = (
                        attention_scores[batch_index] * attention_scores_manipulation[batch_index]
                    )

    attention_probs = softmax_fn(attention_scores, attention_mask)

    if dropout_fn is not None:
        attention_probs = dropout_fn(attention_probs)

    attention_probs = rearrange(attention_probs, "b n s_q s_k -> (b n) s_q s_k")

    hidden_state = torch.bmm(attention_probs.to(dtype=value.dtype), value.transpose(0, 1))

    return hidden_state


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative_seq_lengths_query: torch.Tensor,
    causal: bool,
    query_key_scaling_factor: float,
    dropout_attention_probs: float,
    cumulative_seq_lengths_key: torch.Tensor | None = None,
    local_attention_window_size: Optional[int] = None,
    deterministic_bwd: bool = False,
) -> torch.Tensor:
    """
    Compute attention via flash attention dependency.

    Args:
        query (Tensor [s_q b h])
        key (Tensor [s_k b h])
        value (Tensor [s_k b h])
    Returns:
        Tensor[s_q, b, hp] : context_layer
        Tensor[b np s_q s_k] : attention_probs
    """
    assert flash_attn_varlen_func is not None, "Please install Flash Attention via optimization requirements"

    batch_size = query.shape[1]
    max_seq_length_query = get_max_seq_length(cumulative_seq_lengths_query)
    if cumulative_seq_lengths_key is None:
        max_seq_length_key = max_seq_length_query
        cumulative_seq_lengths_key = cumulative_seq_lengths_query
    else:
        max_seq_length_key = get_max_seq_length(cumulative_seq_lengths_key)

    # reshape into format expected by flash attention [sq, b, np, hn] => [b * sq, np, hn]
    query = rearrange(query, "s_q b n h -> (b s_q) n h")
    key = rearrange(key, "s_k b n h -> (b s_k) n h")
    value = rearrange(value, "s_k b n h -> (b s_k) n h")

    if local_attention_window_size is None:
        local_attention_window_size = -1

    attention_output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cumulative_seq_lengths_query,
        cu_seqlens_k=cumulative_seq_lengths_key,
        max_seqlen_q=max_seq_length_query,
        max_seqlen_k=max_seq_length_key,
        dropout_p=dropout_attention_probs,
        softmax_scale=query_key_scaling_factor,
        causal=causal,
        window_size=(local_attention_window_size, local_attention_window_size),
        deterministic=deterministic_bwd,
    )
    return rearrange(attention_output, "(b sq) np hn -> (b np) sq hn", b=batch_size)


class RelativePositionEmbeddingType(Enum):
    NONE = "none"
    ROTARY = "rotary"
    ROTARY_COMPLEX = "rotary_complex"


class ParallelSelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        masked_softmax_config: MaskedSoftmaxConfig,
        causal: bool = True,
        num_local_attention_heads: int = 0,
        local_attention_window_size: Optional[int] = None,
        scaling_factor: Optional[float] = None,
        dropout_attention_probs: float = 0.0,
        rotary_config: Optional[RotaryConfig] = None,
        relative_position_embedding_type: RelativePositionEmbeddingType = RelativePositionEmbeddingType.ROTARY,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
        bitfit_bias_name: Optional[str] = None,
        lora_config: Optional[LoRaConfig] = None,
        norm_type: NormType = NormType.LAYERNORM,
        key_query_norm: bool = False,
        layernorm_config: Optional[LayerNormConfig] = None,
        qkv_in_one: bool = True,
        num_kv_heads: Optional[int] = None,
        use_matmul: bool = False,
    ) -> None:
        super().__init__()

        assert not (topology is not None and device is not None), "cannot specify both device and topology"
        if topology is None:
            self._device: torch.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._device = topology.device

        assert (
            hidden_size % num_attention_heads == 0
        ), f"hidden size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"

        self.hidden_size_per_attention_head = hidden_size // num_attention_heads

        self.num_attention_heads = num_attention_heads

        self.causal = causal

        self.lora_config = lora_config

        self.use_flash_attention = masked_softmax_config.kernel == MaskedSoftmaxKernel.FLASH_ATTENTION

        self.masked_softmax_config = masked_softmax_config

        self.num_local_attention_heads = num_local_attention_heads
        self.local_attention_window_size = local_attention_window_size

        if self.num_local_attention_heads > 0:
            assert self.use_flash_attention, "local attention is currently only supported, with `flash_attention`."
            assert (
                self.local_attention_window_size is not None
            ), "`local_attention_window_size` needs to be set if `num_local_attention_heads`."

            if self.num_local_attention_heads != self.local_attention_window_size:
                if topology is not None and topology.model_parallel_group.size() > 1:  # type: ignore[union-attr]
                    raise NotImplementedError(
                        "Mixed range attention is currently not supported when using tensor parallelism."
                    )

        if topology is None:
            model_parallel_size = 1
        else:
            model_parallel_size = topology.config.model_parallel_size
        assert self.num_attention_heads % model_parallel_size == 0, (
            f"attention heads ({self.num_attention_heads}) "
            f"must be divisible by model parallel size ({model_parallel_size})"
        )
        self.num_attention_heads_per_partition = self.num_attention_heads // model_parallel_size

        self.dtype = dtype

        # initialize
        self.qkv_in_one = qkv_in_one
        self.num_kv_heads = num_kv_heads
        if self.num_kv_heads:
            assert not self.qkv_in_one, "for a differing number of kv heads, qkv cannot be stored in one"
            self.num_kv_heads_per_partition = self.num_kv_heads // model_parallel_size
            self.num_repeat_kv = self.num_attention_heads_per_partition // self.num_kv_heads_per_partition
        else:
            self.num_kv_heads_per_partition = self.num_attention_heads_per_partition
            self.num_repeat_kv = 1

        if self.lora_config:
            self.lora_merged_state = False
            self.lora_modules = torch.nn.ModuleDict()
            for module_type in self.lora_config.parallel_modules:
                if module_type in [LoRAModuleType.DENSE, LoRAModuleType.QUERY]:
                    repeat_factor = 1
                else:
                    repeat_factor = self.num_repeat_kv

                self.lora_modules[f"{module_type.value}_{self.lora_config.name}"] = ParallelLoRa(
                    in_features=hidden_size,
                    out_features=hidden_size // repeat_factor,
                    rank=self.lora_config.rank,
                    topology=topology,
                    dropout=self.lora_config.dropout,
                    dtype=dtype,
                    lora_module_type=module_type,
                    alpha=self.lora_config.alpha,
                    bias=self.lora_config.bias,
                    kaiming_a=self.lora_config.kaiming_a,
                )

        if self.qkv_in_one:
            self.query_key_value = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=hidden_size * 3,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=init_method,
                bitfit_bias_name=bitfit_bias_name,
                parallel_output=True,
            )

        else:
            self.query = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=init_method,
                bitfit_bias_name=bitfit_bias_name,
                parallel_output=True,
            )
            self.key = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=hidden_size // self.num_repeat_kv,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=init_method,
                bitfit_bias_name=bitfit_bias_name,
                parallel_output=True,
            )
            self.value = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=hidden_size // self.num_repeat_kv,
                bias=bias,
                device=device,
                dtype=dtype,
                topology=topology,
                init_method=init_method,
                bitfit_bias_name=bitfit_bias_name,
                parallel_output=True,
            )
        self.use_matmul = use_matmul

        if scaling_factor is None:
            self.scaling_factor = 1 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.scaling_factor = scaling_factor

        self.rotary_embedding: Optional[Union[RotaryEmbedding, RotaryEmbeddingComplex]] = None
        if relative_position_embedding_type == RelativePositionEmbeddingType.NONE:
            self.rotary_embedding = None
        elif relative_position_embedding_type == RelativePositionEmbeddingType.ROTARY:
            assert rotary_config is not None
            self.rotary_embedding = RotaryEmbedding(
                config=rotary_config,
                device=self._device,
                dtype=dtype,
            )
        elif relative_position_embedding_type == RelativePositionEmbeddingType.ROTARY_COMPLEX:
            assert rotary_config is not None
            self.rotary_embedding = RotaryEmbeddingComplex(
                config=rotary_config,
                device=self._device,
            )
        else:
            raise NotImplementedError

        self.key_query_norm = key_query_norm
        self.topology = topology
        self.norm_query: Optional[Union[LayerNorm, RMSNorm]] = None
        self.norm_key: Optional[Union[LayerNorm, RMSNorm]] = None
        if self.key_query_norm:
            self.norm_query = get_norm(  # type: ignore[operator]
                norm_type=norm_type,
                layernorm_config=layernorm_config,
                dimensions=self.hidden_size_per_attention_head,
                device=self._device,
                dtype=dtype,
                bitfit_bias_name=bitfit_bias_name,
            )
            self.norm_key = get_norm(  # type: ignore[operator]
                norm_type=norm_type,
                layernorm_config=layernorm_config,
                dimensions=self.hidden_size_per_attention_head,
                device=self._device,
                dtype=dtype,
                bitfit_bias_name=bitfit_bias_name,
            )

        self.dropout_attention_probs = dropout_attention_probs
        self.dropout = torch.nn.Dropout(dropout_attention_probs)

        self.dense = RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            topology=topology,
            dtype=dtype,
            bitfit_bias_name=bitfit_bias_name,
            init_method=init_method,
            parallel_input=True,
            parallel_output=(self.topology.config.sequence_parallel if self.topology is not None else False),
        )

        self.masked_softmax = MaskedSoftmax(config=masked_softmax_config)

        self.cache: Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = dict()

    def forward(
        self,
        x: torch.Tensor,
        cumulative_seq_lengths: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cumulative_seq_lengths_key: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        reset_cache: bool = False,
        cache_index: int = 0,
        attention_scores_manipulation: Optional[torch.Tensor] = None,
        attentions_score_manipulation_log_additive: Union[bool, List[bool]] = True,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        # get query, key and value
        # attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        if self.qkv_in_one:
            query_key_value_mat = self.query_key_value(x)
            query_key_value_mat = rearrange(
                query_key_value_mat,
                "b sq (np hn) -> sq b np hn",
                np=self.num_attention_heads_per_partition,
            )

            (query, key, value) = split_tensor_along_last_dim(query_key_value_mat, 3)

        else:
            # [3*hidden_size, hidden_size] -> 3*[hidden_size, hidden_size]
            query = rearrange(
                self.query(x),
                "b sq (np hn) -> sq b np hn",
                np=self.num_attention_heads_per_partition,
            )
            key = rearrange(
                self.key(x),
                "b sq (np hn) -> sq b np hn",
                np=self.num_kv_heads_per_partition,
            )
            value = rearrange(
                self.value(x),
                "b sq (np hn) -> sq b np hn",
                np=self.num_kv_heads_per_partition,
            )

        if self.lora_config is not None and not self.lora_merged_state:
            query, key, value = self.apply_lora(x, query, key, value)

        # apply query and key norm
        if self.key_query_norm:
            assert self.topology is not None
            assert self.norm_query is not None
            assert self.norm_key is not None

            # in order not to have parameters of the norm diverge
            # all_concat and all_shard is used to reduce and split across model parallel
            # the dimension does not really matter as long it is not the last, but we take the head dim
            query = all_concat(query, dim=2, topology=self.topology)
            query = self.norm_query(query)
            query = all_shard(query, dim=2, topology=self.topology)

            key = all_concat(key, dim=2, topology=self.topology)
            key = self.norm_key(key)
            key = all_shard(key, dim=2, topology=self.topology)

        # apply rotary embedding
        if self.rotary_embedding is not None:
            if position_ids is not None:
                position_ids = rearrange(position_ids, "b sq -> sq b")
            query, key = self.rotary_embedding(
                query=query,
                key=key,
                query_position_ids=position_ids,
                key_position_ids=position_ids,
            )

        def get_trivial_cumulative_seq_lengths(activations: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0, activations.shape[0]], device=activations.device, dtype=torch.int32)

        if use_cache:
            if not self.causal:
                raise ValueError("KV caching is only supported for causal attention.")
            assert (
                key.shape[1] == 1 and value.shape[1] == 1 and query.shape[1] == 1
            ), f"KV caching is only supported for batch size 1, got {key.shape[1]}, {value.shape[1]}, {query.shape[1]}"
            assert torch.equal(
                cumulative_seq_lengths, get_trivial_cumulative_seq_lengths(query)
            ), "sequence packing is not supported with cache"
            if reset_cache:
                self.cache[cache_index] = (key, value)
            else:
                past_key, past_value = self.cache[cache_index]
                assert past_key is not None
                assert past_value is not None
                key = torch.cat((past_key, key), dim=0)
                value = torch.cat((past_value, value), dim=0)
                self.cache[cache_index] = (key, value)
            cumulative_seq_lengths_key = get_trivial_cumulative_seq_lengths(key)
        else:
            if reset_cache:
                self.cache[cache_index] = (None, None)

        # Normal Flash attention and Simple Local Attention
        # This If Block is used when we have flash Attention without any local attention
        # (num_local_attention_heads is zero or smaller)
        # And this Block is called when we have local attention, but on all attention heads
        # (num_local_attention_heads is equal to num_attention_heads)
        if self.use_flash_attention and (
            self.num_local_attention_heads <= 0
            or (self.num_local_attention_heads > 0 and self.num_local_attention_heads == self.num_attention_heads)
        ):
            hidden_state = flash_attention(
                query=query,
                key=key,
                value=value,
                cumulative_seq_lengths_query=cumulative_seq_lengths,
                cumulative_seq_lengths_key=cumulative_seq_lengths_key,
                causal=self.causal,
                query_key_scaling_factor=self.scaling_factor,
                dropout_attention_probs=self.dropout_attention_probs,
                local_attention_window_size=self.local_attention_window_size,
                deterministic_bwd=self.masked_softmax_config.deterministic_flash_attn_bwd,
            )

            hidden_state = rearrange(hidden_state, "(b np) sq hn -> b sq (np hn)", b=batch_size)

        # Mixed Range Attention
        elif (
            self.use_flash_attention
            and self.num_local_attention_heads > 0
            and self.num_local_attention_heads != self.num_attention_heads
        ):
            # We have to repeat here in order to properly slice the attention heads in to local and global heads.
            key = repeat_kv(key, self.num_repeat_kv)  # (bs, seqlen, n_heads, head_dim)
            value = repeat_kv(value, self.num_repeat_kv)  # (bs, seqlen, n_heads, head_dim)

            # Take the first num_local_attention_heads heads and use them for local attention.
            local_attention_hidden_state = flash_attention(
                query=query[:, :, : self.num_local_attention_heads],
                key=key[:, :, : self.num_local_attention_heads],
                value=value[:, :, : self.num_local_attention_heads],
                cumulative_seq_lengths_query=cumulative_seq_lengths,
                cumulative_seq_lengths_key=cumulative_seq_lengths_key,
                causal=self.causal,
                query_key_scaling_factor=self.scaling_factor,
                dropout_attention_probs=self.dropout_attention_probs,
                local_attention_window_size=self.local_attention_window_size,
                deterministic_bwd=self.masked_softmax_config.deterministic_flash_attn_bwd,
            )

            # Take the remaining attention heads and use them for global attention.
            global_attention_hidden_state = flash_attention(
                query=query[:, :, self.num_local_attention_heads :],
                key=key[:, :, self.num_local_attention_heads :],
                value=value[:, :, self.num_local_attention_heads :],
                cumulative_seq_lengths_query=cumulative_seq_lengths,
                cumulative_seq_lengths_key=cumulative_seq_lengths_key,
                causal=self.causal,
                query_key_scaling_factor=self.scaling_factor,
                dropout_attention_probs=self.dropout_attention_probs,
                deterministic_bwd=self.masked_softmax_config.deterministic_flash_attn_bwd,
            )

            # Reshape both hidden states for concatenation.
            local_attention_hidden_state = rearrange(
                local_attention_hidden_state, "(b np) sq hn -> b sq np hn", b=batch_size
            )

            global_attention_hidden_state = rearrange(
                global_attention_hidden_state,
                "(b np) sq hn -> b sq np hn",
                b=batch_size,
            )

            hidden_state = torch.cat([local_attention_hidden_state, global_attention_hidden_state], dim=2)
            hidden_state = rearrange(hidden_state, "b sq np hn -> b sq (np hn)")

        # Torch Attention
        else:
            key = repeat_kv(key, self.num_repeat_kv)  # (bs, seqlen, n_local_heads, head_dim)
            value = repeat_kv(value, self.num_repeat_kv)  # (bs, seqlen, n_local_heads, head_dim)

            hidden_state = multi_head_attention(
                query=query,
                key=key,
                value=value,
                cumulative_seq_lengths=cumulative_seq_lengths,
                causal=self.causal,
                query_key_scaling_factor=self.scaling_factor,
                softmax_fn=self.masked_softmax,
                dropout_fn=self.dropout,
                attention_scores_manipulation=attention_scores_manipulation,
                attentions_score_manipulation_log_additive=attentions_score_manipulation_log_additive,
                use_matmul=self.use_matmul,
            )

            hidden_state = rearrange(hidden_state, "(b np) sq hn -> b sq (np hn)", b=batch_size)

        if self.lora_config and not self.lora_merged_state:
            if LoRAModuleType.DENSE in self.lora_config.parallel_modules:
                assert self.topology
                dense_reduced_input = all_concat(hidden_state, dim=-1, topology=self.topology)
                dense_lora_parallel_out = self.lora_modules[f"dense_{self.lora_config.name}"](dense_reduced_input)

        # apply final dense
        hidden_state_out = self.dense(hidden_state)

        if self.lora_config and not self.lora_merged_state:
            if LoRAModuleType.DENSE in self.lora_config.parallel_modules:
                hidden_state_out = torch.add(hidden_state_out, dense_lora_parallel_out)

        # scatter to sequence parallel
        if self.topology is not None and self.topology.config.sequence_parallel:
            hidden_state_out = all_reduce_scatter_to_sequence_parallel(hidden_state_out, self.topology)

        return hidden_state_out

    def apply_lora(
        self, x: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> List[torch.Tensor]:
        assert self.lora_config, "LoRA Config needs to be set in order to apply lora modules"
        head_count_config = {
            "query": self.num_attention_heads_per_partition,
            "key": (self.num_kv_heads_per_partition if not self.qkv_in_one else self.num_attention_heads_per_partition),
            "value": (
                self.num_kv_heads_per_partition if not self.qkv_in_one else self.num_attention_heads_per_partition
            ),
        }

        # Iterate over the LoRa modules and apply the transformations
        for lora_module_key, lora_module in self.lora_modules.items():
            # Skip the output ('dense') module; it's handled separately
            if lora_module_key == f"dense_{self.lora_config.name}":
                continue

            # Apply the LoRa module to the input
            lora_parallel_out = lora_module(x)

            # Rearrange output according to the LoRa module type
            np = head_count_config[
                lora_module.lora_module_type.value
            ]  # Get the number of heads/partition for the module
            lora_parallel_out = rearrange(lora_parallel_out, "b sq (np hn) -> sq b np hn", np=np)

            # Add the LoRa adjustment to the appropriate tensor (query, key, or value)
            if lora_module.lora_module_type == LoRAModuleType.QUERY:
                query = torch.add(query, lora_parallel_out)
            elif lora_module.lora_module_type == LoRAModuleType.KEY:
                key = torch.add(key, lora_parallel_out)
            elif lora_module.lora_module_type == LoRAModuleType.VALUE:
                value = torch.add(value, lora_parallel_out)

        return [query, key, value]

    def _get_delta_one_q_k_v(self) -> torch.Tensor:
        assert self.lora_modules and self.lora_config, "LoRa configuration or modules are not properly specified."

        weight_shape = self.query_key_value.weight.shape
        delta_accumulator = torch.zeros(weight_shape, dtype=self.dtype).to(self._device)
        chunks = list(torch.chunk(delta_accumulator, 3, dim=0))

        module_mapping = {
            f"query_{self.lora_config.name}": 0,
            f"key_{self.lora_config.name}": 1,
            f"value_{self.lora_config.name}": 2,
        }

        for key, module in self.lora_modules.items():
            if key in module_mapping:
                delta = module.get_delta_weights()
                chunks[module_mapping[key]] += delta

        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        assert self.lora_config and self.lora_modules, "Merge of LoRa weights called without proper configuration."

        if self.qkv_in_one:
            relevant_keys = {
                f"{m.value}_{self.lora_config.name}"
                for m in [
                    LoRAModuleType.KEY,
                    LoRAModuleType.VALUE,
                    LoRAModuleType.QUERY,
                ]
            }
            if relevant_keys.intersection(self.lora_modules.keys()):
                self.query_key_value.weight.data += self._get_delta_one_q_k_v()
        else:
            for key, module in self.lora_modules.items():
                key = key.split("_")[0]
                if not key == "dense":
                    target_attr = getattr(self, key)
                    target_attr.weight += module.get_delta_weights()

        if LoRAModuleType.DENSE in self.lora_config.parallel_modules:
            dense_module: ParallelLoRa = self.lora_modules[f"dense_{self.lora_config.name}"]
            self.dense.weight.data += dense_module.get_delta_weights()

        del self.lora_modules
        gc.collect()
        torch.cuda.empty_cache()

        self.lora_merged_state = True
