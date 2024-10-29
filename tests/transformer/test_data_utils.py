import pytest
import torch

from scaling.transformer.data.utils import (
    add_cumulative_seq_lengths_padding,
    get_cumulative_seq_lengths,
    get_position_ids,
    remove_cumulative_seq_lengths_padding,
)


@pytest.fixture
def input_token_ids():
    return torch.tensor(
        [
            [1, 2, 3, 0, 1, 2],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 0],
        ]
    )


@pytest.mark.transformer
def test_get_cumulative_seq_lengths_with_reset(input_token_ids: torch.Tensor):
    cumulative_seq_lengths = get_cumulative_seq_lengths(input_token_ids)
    assert torch.equal(cumulative_seq_lengths, torch.tensor([0, 4, 6, 12, 18]))


@pytest.mark.transformer
def test_get_cumulative_seq_lengths_without_reset(input_token_ids: torch.Tensor):
    cumulative_seq_lengths = get_cumulative_seq_lengths(input_token_ids, reset_attention_mask=False)
    assert torch.equal(cumulative_seq_lengths, torch.tensor([0, 6, 12, 18]))


@pytest.mark.transformer
def test_get_position_ids_with_reset(input_token_ids: torch.Tensor):
    position_ids = get_position_ids(input_token_ids=input_token_ids, reset_position_ids=True)
    assert torch.equal(
        position_ids,
        torch.tensor(
            [
                [0, 1, 2, 3, 0, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ]
        ),
    )


@pytest.mark.transformer
def test_get_position_ids_without_reset(input_token_ids: torch.Tensor):
    position_ids = get_position_ids(input_token_ids=input_token_ids, reset_position_ids=False)
    assert torch.equal(
        position_ids,
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ]
        ),
    )


@pytest.mark.transformer
def test_add_cumulative_seq_lengths_padding():
    cumulative_seq_lengths = torch.tensor([0, 4, 8, 12])
    padded_cumulative_seq_lengths = add_cumulative_seq_lengths_padding(cumulative_seq_lengths, pad_to=16)
    expected_padded_cumulative_seq_lengths = torch.tensor([0, 4, 8, 12] + [-1] * 12)
    assert torch.equal(padded_cumulative_seq_lengths, expected_padded_cumulative_seq_lengths)


@pytest.mark.transformer
def test_add_cumulative_seq_lengths_padding_works_with_no_padding():
    cumulative_seq_lengths = torch.tensor([0, 4, 8, 12])
    padded_cumulative_seq_lengths = add_cumulative_seq_lengths_padding(cumulative_seq_lengths, 4)
    assert torch.equal(padded_cumulative_seq_lengths, cumulative_seq_lengths)


@pytest.mark.transformer
def test_remove_cumulative_seq_lengths_padding():
    cumulative_seq_lengths = torch.tensor([0, 4, 8, 12])
    padded_cumulative_seq_lengths = add_cumulative_seq_lengths_padding(cumulative_seq_lengths, pad_to=16)
    output_cumulative_seq_lengths = remove_cumulative_seq_lengths_padding(padded_cumulative_seq_lengths)
    assert output_cumulative_seq_lengths is not None
    assert torch.equal(output_cumulative_seq_lengths, cumulative_seq_lengths)
