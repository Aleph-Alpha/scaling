from typing import Optional

import pytest
import torch

from scaling.core import MaskedSoftmaxConfig, MaskedSoftmaxKernel, ParallelSelfAttention, RelativePositionEmbeddingType


@pytest.mark.parametrize("num_kv_heads", [None, 2, 4, 8, 16])
@pytest.mark.parametrize("num_local_attention_heads", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("local_attention_window_size", [64, 256, 1024])
def test_flash_attention_with_group_query_attention_output(
    num_kv_heads: Optional[int],
    num_local_attention_heads: int,
    local_attention_window_size: Optional[int],
):
    """
    Comparing outputs of Flash Attention and
    non Flash Attention parallel self attention
    with Group query attention and less kv heads
    """
    batch_size = 2
    seq_length = 2048
    hidden_size = 128
    num_attention_heads = 16

    # Input: [batch_size, seq_length, hidden_size]
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input = torch.rand(batch_size, seq_length, hidden_size, dtype=torch.float16).cuda()
    cumulative_seq_lengths = torch.arange(0, batch_size * seq_length + 1, seq_length, dtype=torch.int32).cuda()

    # Initialize parallel self attention with using flash attention
    flash_attn = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=MaskedSoftmaxKernel.FLASH_ATTENTION),
        dropout_attention_probs=0.0,
        rotary_config=None,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
        bias=False,
        dtype=torch.float16,
        qkv_in_one=False,
        num_kv_heads=num_kv_heads,
        num_local_attention_heads=num_local_attention_heads,
        local_attention_window_size=local_attention_window_size,
    )

    flash_attn_output = flash_attn(input, cumulative_seq_lengths, position_ids=None)

    assert not torch.any(torch.isnan(flash_attn_output))
