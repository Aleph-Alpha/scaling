import pytest
import torch

from scaling.core.nn.attention import (
    cumulative_seq_lengths_to_dense_attention_mask,
    get_max_seq_length,
)


@pytest.fixture
def cumulative_seq_lengths() -> torch.Tensor:
    return torch.tensor([0, 2, 4, 6, 9, 12, 18])


@pytest.fixture
def seq_length_per_batch_item() -> int:
    return 6


def test_cumulative_seq_lengths_to_dense_attention_mask_causal(
    cumulative_seq_lengths: torch.Tensor, seq_length_per_batch_item: int
) -> None:
    attention_mask = cumulative_seq_lengths_to_dense_attention_mask(
        cumulative_seq_lengths, seq_length_per_batch_item, causal=True
    )
    assert attention_mask.size() == (3, 1, 6, 6)  # = (batch_size, num_heads, seq_length, seq_length)
    expected_result = ~torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(attention_mask[:, 0, :, :], expected_result)


def test_cumulative_seq_lengths_to_dense_attention_mask_bidirectional(
    cumulative_seq_lengths: torch.Tensor, seq_length_per_batch_item: int
) -> None:
    attention_mask = cumulative_seq_lengths_to_dense_attention_mask(
        cumulative_seq_lengths, seq_length_per_batch_item, causal=False
    )
    assert attention_mask.size() == (3, 1, 6, 6)  # = (batch_size, num_heads, seq_length, seq_length)
    expected_result = ~torch.tensor(
        [
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(attention_mask[:, 0, :, :], expected_result)


def test_get_max_seq_length_on_example(cumulative_seq_lengths: torch.Tensor) -> None:
    max_seq_length = get_max_seq_length(cumulative_seq_lengths)
    assert max_seq_length == 6
