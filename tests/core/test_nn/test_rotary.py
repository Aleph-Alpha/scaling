import pytest
import torch

from scaling.core import RotaryConfig, RotaryEmbedding, RotaryEmbeddingComplex


@pytest.mark.transformer_module
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [10, 17])
@pytest.mark.parametrize("hidden_size_per_attention_head", [2, 8, 16])
@pytest.mark.parametrize("attention_heads", [1, 2, 4])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("batch_size", [2, 7])
def test_rotary(
    dtype: torch.dtype,
    seq_len: int,
    hidden_size_per_attention_head: int,
    attention_heads: int,
    device: torch.device,
    batch_size: int,
):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU available")
    config = RotaryConfig(dimensions=hidden_size_per_attention_head, max_seq_length=seq_len, base=10000)
    rotary_embedding = RotaryEmbedding(
        config=config,
        device=device,
        dtype=dtype,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    query = torch.randn(
        (seq_len, batch_size, attention_heads, hidden_size_per_attention_head),
        dtype=dtype,
        device=device,
    )
    query_position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .repeat(batch_size)
        .reshape((batch_size, seq_len))
        .transpose(1, 0)
    )
    key = torch.randn(
        (seq_len, batch_size, attention_heads, hidden_size_per_attention_head),
        dtype=dtype,
        device=device,
    )
    key_position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .repeat(batch_size)
        .reshape((batch_size, seq_len))
        .transpose(1, 0)
    )

    # embed
    query_rot, key_rot = rotary_embedding(query=query, key=key)
    assert not (query_rot == query).all(), "query did not change after rotary emb"
    assert not (key_rot == key).all(), "key did not change after rotary emb"

    # embed with position ids
    query_rot_position_ids, key_rot_position_ids = rotary_embedding(
        query=query,
        key=key,
        query_position_ids=query_position_ids,
        key_position_ids=key_position_ids,
    )
    assert (query_rot == query_rot_position_ids).all(), "not the same result with and without position ids for query"
    assert (key_rot == key_rot_position_ids).all(), "not the same result with and without position ids for key"

    # test offset
    query_rot_offset, key_rot_offset = rotary_embedding(
        query=query[1:, :, :],
        query_position_ids=query_position_ids[1:],
        key=key[1:, :, :],
        key_position_ids=key_position_ids[1:],
    )
    assert (query_rot[1:, :, :] == query_rot_offset).all(), "query offset not applied correctly"
    assert (key_rot[1:, :, :] == key_rot_offset).all(), "query offset not applied correctly"


@pytest.mark.transformer_module
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("seq_len", [10, 17])
@pytest.mark.parametrize("hidden_size_per_attention_head", [2, 8, 16])
@pytest.mark.parametrize("attention_heads", [1, 2, 4])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("batch_size", [2, 7])
def test_rotary_llama(
    dtype: torch.dtype,
    seq_len: int,
    hidden_size_per_attention_head: int,
    attention_heads: int,
    device: torch.device,
    batch_size: int,
):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU available")
    config = RotaryConfig(dimensions=hidden_size_per_attention_head, max_seq_length=seq_len, base=10000.0)
    rotary_embedding = RotaryEmbeddingComplex(
        config=config,
        device=device,
    )
    rotary_baseline = RotaryEmbedding(
        config=config,
        device=device,
        dtype=dtype,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    query = torch.ones(
        (seq_len, batch_size, attention_heads, hidden_size_per_attention_head),
        dtype=dtype,
        device=device,
    )
    query_position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .repeat(batch_size)
        .reshape((batch_size, seq_len))
        .transpose(1, 0)
    )
    key = torch.ones(
        (seq_len, batch_size, attention_heads, hidden_size_per_attention_head),
        dtype=dtype,
        device=device,
    )
    key_position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .repeat(batch_size)
        .reshape((batch_size, seq_len))
        .transpose(1, 0)
    )

    # embed
    query_rot, key_rot = rotary_embedding(query=query, key=key)
    assert not (query_rot == query).all(), "query did not change after rotary emb"
    assert not (key_rot == key).all(), "key did not change after rotary emb"

    # compare to baseline
    query_rot_baseline, key_rot_baseline = rotary_baseline(query=query, key=key)

    # translation between formats
    query_rot_baseline_translated = torch.zeros_like(query_rot_baseline)
    key_rot_baseline_translated = torch.zeros_like(key_rot_baseline)
    keys_src = torch.arange(0, query_rot_baseline.shape[-1] // 2) * 2
    query_rot_baseline_translated[:, :, :, keys_src] = query_rot_baseline[:, :, :, : query_rot_baseline.shape[-1] // 2]
    key_rot_baseline_translated[:, :, :, keys_src] = key_rot_baseline[:, :, :, : key_rot_baseline.shape[-1] // 2]
    keys_src = keys_src + 1
    query_rot_baseline_translated[:, :, :, keys_src] = query_rot_baseline[:, :, :, query_rot_baseline.shape[-1] // 2 :]
    key_rot_baseline_translated[:, :, :, keys_src] = key_rot_baseline[:, :, :, key_rot_baseline.shape[-1] // 2 :]

    assert torch.isclose(query_rot, query_rot_baseline_translated).all(), "rotary for query does not match baseline"
    assert torch.isclose(key_rot, key_rot_baseline_translated).all(), "rotary for key does not match baseline"

    # embed with position ids
    query_rot_position_ids, key_rot_position_ids = rotary_embedding(
        query=query,
        key=key,
        query_position_ids=query_position_ids,
        key_position_ids=key_position_ids,
    )
    assert (query_rot == query_rot_position_ids).all(), "not the same result with and without position ids for query"
    assert (key_rot == key_rot_position_ids).all(), "not the same result with and without position ids for key"

    # test offset
    query_rot_offset, key_rot_offset = rotary_embedding(
        query=query[1:, :, :],
        query_position_ids=query_position_ids[1:],
        key=key[1:, :, :],
        key_position_ids=key_position_ids[1:],
    )
    assert (query_rot[1:, :, :] == query_rot_offset).all(), "query offset not applied correctly"
    assert (key_rot[1:, :, :] == key_rot_offset).all(), "query offset not applied correctly"
