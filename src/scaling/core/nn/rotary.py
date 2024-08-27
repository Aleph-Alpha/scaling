from typing import Optional

import torch
from einops import rearrange, repeat

from scaling.core.nn.rotary_config import RotaryConfig


def vector_gather(vectors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers (batched) vectors according to indices.
    """
    vectors = repeat(vectors, "sq b nh d -> sq b B nh d", B=indices.shape[1]).squeeze(1)
    indices = repeat(
        indices,
        "sq b -> sq b nh d",
        nh=vectors.shape[-2],
        d=vectors.shape[-1],
    )

    out = torch.gather(vectors, dim=0, index=indices)

    return out


def vector_gather_complex(vectors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers (batched) vectors according to indices.
    """
    vectors = repeat(vectors, "sq d -> sq B nh d", B=indices.shape[1], nh=1)
    indices = repeat(
        indices,
        "sq b -> sq b nh d",
        nh=1,
        d=vectors.shape[-1],
    )

    out = torch.gather(vectors, dim=0, index=indices)

    out = rearrange(out, "sq b nh hh -> b sq nh hh")

    return out


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    theta = float(theta)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)).to(device)
    t = torch.arange(end, device=device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(device)


def reshape_complex_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape[0] == x.shape[1]
    assert freqs_cis.shape[1] == x.shape[-1]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_complex_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    query_position_ids: Optional[torch.Tensor],
    key_position_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    if query_position_ids is None:
        freqs_cis_q = reshape_complex_for_broadcast(freqs_cis, xq_complex)
    else:
        freqs_cis_q = vector_gather_complex(freqs_cis, query_position_ids)

    if key_position_ids is None:
        freqs_cis_k = reshape_complex_for_broadcast(freqs_cis, xq_complex)
    else:
        freqs_cis_k = vector_gather_complex(freqs_cis, key_position_ids)

    xq_out = torch.view_as_real(xq_complex * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _pre_calculate_cos_sin(
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seq_len: int,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(device).float()
    cos = torch.reshape(emb.cos(), (seq_len, 1, 1, dim))
    sin = torch.reshape(emb.sin(), (seq_len, 1, 1, dim))
    if dtype != torch.float32:
        cos = cos.type(dtype)
        sin = sin.type(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


# @torch.jit.script
def apply_rotary_pos_emb(
    query: torch.Tensor,
    query_position_ids: Optional[torch.Tensor],
    key: torch.Tensor,
    key_position_ids: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_position_ids is None:
        cos_q = cos
        sin_q = sin
    else:
        cos_q = vector_gather(cos, query_position_ids)
        sin_q = vector_gather(sin, query_position_ids)

    if key_position_ids is None:
        cos_k = cos
        sin_k = sin
    else:
        cos_k = vector_gather(cos, key_position_ids)
        sin_k = vector_gather(sin, key_position_ids)

    return (query * cos_q) + (rotate_half(query) * sin_q), (key * cos_k) + (rotate_half(key) * sin_k)


class RotaryEmbedding(torch.nn.Module):
    """
    Relative rotary position embedding based on
    * RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
    * Rotary Embeddings: A Relative Revolution (https://blog.eleuther.ai/rotary-embeddings/)
    """

    def __init__(
        self,
        config: RotaryConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert config.dimensions > 1, "RotaryEmbedding cannot use `dim` == 1, this results in weird reshape errors"
        self.inv_freq = (
            1.0
            / (config.base ** (torch.arange(0, config.dimensions, 2).float() / config.dimensions).clone().detach())
            .clone()
            .detach()
        )
        cos, sin = _pre_calculate_cos_sin(
            dim=config.dimensions,
            dtype=dtype,
            device=device,
            seq_len=config.max_seq_length,
            inv_freq=self.inv_freq,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.dimensions = config.dimensions

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: Optional[torch.Tensor] = None,
        key_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[-1] != self.dimensions or key.shape[-1] != self.dimensions:
            # We are in the case where we apply rotary to subset of the dimensions
            assert query.shape[-1] > self.dimensions, f"query_dims={query.shape[-1]} rotary_dims={self.dimensions}"
            assert key.shape[-1] > self.dimensions, f"key_dims={key.shape[-1]} rotary_dims={self.dimensions}"

            queries_with_rot = query[..., : self.dimensions]
            queries_without_rot = query[..., self.dimensions :]
            keys_with_rot = key[..., : self.dimensions]
            keys_without_rot = key[..., self.dimensions :]

            queries_with_rot, keys_with_rot = apply_rotary_pos_emb(
                query=queries_with_rot,
                key=keys_with_rot,
                cos=self.cos,
                sin=self.sin,
                query_position_ids=query_position_ids,
                key_position_ids=key_position_ids,
            )

            query = torch.cat((queries_with_rot, queries_without_rot), dim=-1)
            key = torch.cat((keys_with_rot, keys_without_rot), dim=-1)
        else:
            query, key = apply_rotary_pos_emb(
                query=query,
                key=key,
                cos=self.cos,
                sin=self.sin,
                query_position_ids=query_position_ids,
                key_position_ids=key_position_ids,
            )
        return query, key


class RotaryEmbeddingComplex(torch.nn.Module):
    """
    Relative rotary position embedding based on
    * RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
    * Rotary Embeddings: A Relative Revolution (https://blog.eleuther.ai/rotary-embeddings/)
    """

    def __init__(
        self,
        config: RotaryConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert config.dimensions > 1, "RotaryEmbedding cannot use `dim` == 1, this results in weird reshape errors"

        freqs_cis = precompute_freqs_cis(
            dim=config.dimensions,
            end=config.max_seq_length,
            theta=config.base,
            device=device,
        )
        # Store real and imaginary in separate buffers for correct type casting.
        self.register_buffer("freqs_cis_real", freqs_cis.real, persistent=False)
        self.register_buffer("freqs_cis_imag", freqs_cis.imag, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: Optional[torch.Tensor] = None,
        key_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key = apply_complex_rotary_emb(
            xq=rearrange(query, "sq b nh hh -> b sq nh hh"),
            xk=rearrange(key, "sq b nh hh -> b sq nh hh"),
            freqs_cis=torch.complex(self.freqs_cis_real.float(), self.freqs_cis_imag.float()),
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        return rearrange(query, "b sq nh hh -> sq b nh hh"), rearrange(key, "b sq nh hh -> sq b nh hh")
