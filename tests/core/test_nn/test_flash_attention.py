import pytest
import torch

from scaling.core import MaskedSoftmaxConfig, MaskedSoftmaxKernel, ParallelSelfAttention, RelativePositionEmbeddingType


@pytest.mark.transformer_flash_attn
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_output(causal):
    """
    Comparing outputs of Flash Attention and non Flash Attention parallel self attention which should be the same
    """
    batch_size = 2
    seq_length = 64
    hidden_size = 128
    num_attention_heads = 4

    # Initialize parallel self attention without using flash attention
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    attn = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=MaskedSoftmaxKernel.TORCH),
        causal=causal,
        dropout_attention_probs=0.0,
        rotary_config=None,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
        bias=False,
        dtype=torch.float16,
    )

    # Input: [batch_size, seq_length, hidden_size]
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input = torch.rand(batch_size, seq_length, hidden_size, dtype=torch.float16).cuda()
    cumulative_seq_lengths = torch.tensor([0, 32, 64, 128], dtype=torch.int32).cuda()
    attn_output = attn(input, cumulative_seq_lengths, position_ids=None)

    # Initialize parallel self attention with using flash attention
    flash_attn = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=MaskedSoftmaxKernel.FLASH_ATTENTION),
        causal=causal,
        dropout_attention_probs=0.0,
        rotary_config=None,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
        bias=False,
        dtype=torch.float16,
    )

    # Overwrite weight initializations
    state_dict = attn.state_dict().copy()
    flash_attn.load_state_dict(state_dict)

    flash_attn_output = flash_attn(input, cumulative_seq_lengths, position_ids=None)

    assert torch.allclose(
        attn_output, flash_attn_output, atol=1e-3
    ), "Outputs of Flash Attention and non Flash Attention is not equal"


@pytest.mark.transformer_flash_attn
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_with_group_query_attention_output(causal):
    """
    Comparing outputs of Flash Attention and
    non Flash Attention parallel self attention
    with Group query attention and less kv heads
    """
    batch_size = 2
    seq_length = 64
    hidden_size = 128
    num_attention_heads = 4
    num_kv_heads = 2

    # Initialize parallel self attention without using flash attention
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    attn = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=MaskedSoftmaxKernel.TORCH),
        causal=causal,
        dropout_attention_probs=0.0,
        rotary_config=None,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
        bias=False,
        dtype=torch.float16,
        qkv_in_one=False,
        num_kv_heads=num_kv_heads,
    )

    # Input: [batch_size, seq_length, hidden_size]
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input = torch.rand(batch_size, seq_length, hidden_size, dtype=torch.float16).cuda()
    cumulative_seq_lengths = torch.tensor([0, 32, 64, 128], dtype=torch.int32).cuda()
    attn_output = attn(input, cumulative_seq_lengths, position_ids=None)

    # Initialize parallel self attention with using flash attention
    flash_attn = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        masked_softmax_config=MaskedSoftmaxConfig(kernel=MaskedSoftmaxKernel.FLASH_ATTENTION),
        causal=causal,
        dropout_attention_probs=0.0,
        rotary_config=None,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
        bias=False,
        dtype=torch.float16,
        qkv_in_one=False,
        num_kv_heads=num_kv_heads,
    )

    # Overwrite weight initializations
    state_dict = attn.state_dict().copy()
    flash_attn.load_state_dict(state_dict)

    flash_attn_output = flash_attn(input, cumulative_seq_lengths, position_ids=None)

    assert torch.allclose(
        attn_output, flash_attn_output, atol=1e-3
    ), "Outputs of Flash Attention and non Flash Attention is not equal"
