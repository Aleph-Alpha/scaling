import torch


def add_cumulative_seq_lengths_padding(
    cumulative_seq_lengths: torch.Tensor, pad_to: int, padding_value: int = -1
) -> torch.Tensor:
    """
    Adds padding to the cumulative sequence lengths tensor.

    Args:
        cumulative_seq_lengths (torch.Tensor): The tensor containing the cumulative sequence lengths.
        pad_to (int): The desired length of the padded tensor.

    Returns:
        torch.Tensor: The tensor with padding added to the end.
    """
    assert pad_to >= len(cumulative_seq_lengths)
    num_padding_elements = pad_to - len(cumulative_seq_lengths)
    padding = torch.full(
        (num_padding_elements,), padding_value, dtype=torch.int32, device=cumulative_seq_lengths.device
    )
    return torch.cat((cumulative_seq_lengths, padding))


def remove_cumulative_seq_lengths_padding(
    cumulative_seq_lengths: torch.Tensor, padding_value: int = -1
) -> torch.Tensor:
    """
    Removes padding from the cumulative sequence lengths tensor.

    Args:
        cumulative_seq_lengths (torch.Tensor): The tensor containing the cumulative sequence lengths.

    Returns:
        torch.Tensor: The tensor with padding removed from the end.
    """
    return cumulative_seq_lengths[cumulative_seq_lengths != padding_value]


def get_cumulative_seq_lengths(
    input_token_ids: torch.Tensor,
    reset_attention_mask: bool = True,
    eod_token: int = 0,  # TODO hardcoded for transformer
) -> torch.Tensor:
    """
    Compute the cumulative sequence lengths for a given input tensor.

    Args:
        input_token_ids (torch.Tensor): The input tensor containing token IDs.
        reset_attention_mask (bool, optional): Whether to reset the attention mask
            after a sequence specified by the eod_token. Defaults to True.
        eod_token (int, optional): The end-of-document token. Defaults to 0.

    Returns:
        torch.Tensor: The cumulative sequence lengths tensor.

    """
    micro_batch_size, seq_length = input_token_ids.size()

    if reset_attention_mask:
        # Compute idx of tokens that are eod or the last token in sequence
        is_last_token = torch.arange(seq_length, device=input_token_ids.device) == (seq_length - 1)
        is_eod_token = input_token_ids == eod_token
        batch_idx, seq_idx = torch.where(torch.logical_or(is_eod_token, is_last_token))
        cumulative_seq_lengths = batch_idx * seq_length + seq_idx + 1
        cumulative_seq_lengths = cumulative_seq_lengths.to(torch.int32)
        zero = torch.tensor([0], dtype=torch.int32, device=input_token_ids.device)
        cumulative_seq_lengths = torch.cat([zero, cumulative_seq_lengths], dim=0)
    else:
        cumulative_seq_lengths = torch.arange(
            0, micro_batch_size * seq_length + 1, seq_length, dtype=torch.int32, device=input_token_ids.device
        )

    return cumulative_seq_lengths


def get_position_ids(
    input_token_ids: torch.Tensor,
    reset_position_ids: bool = True,
    eod_token: int = 0,  # TODO hardcoded for transformer
) -> torch.Tensor:
    """
    Generate running position ids for input token ids.

    Args:
        input_token_ids (torch.Tensor): The input token ids.
        reset_position_ids (bool, optional): Whether to reset the position after the eod token. Defaults to True.
        eod_token (int, optional): End-of-document token. Defaults to 0.

    Returns:
        torch.Tensor: The position IDs.

    """
    micro_batch_size, seq_length = input_token_ids.size()
    if reset_position_ids:
        position_ids = torch.zeros_like(input_token_ids)
        for b in range(micro_batch_size):
            last_token = torch.arange(seq_length, device=input_token_ids.device) == (seq_length - 1)
            (seq_idx,) = torch.where(torch.logical_or(input_token_ids[b, :] == eod_token, last_token))
            start = 0
            for pos in seq_idx:
                end = int(pos) + 1
                position_ids[b, start:end] = torch.arange(end - start)
                start = end
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_token_ids)
    return position_ids
