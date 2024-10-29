import pytest
import torch

from scaling.core import (
    MaskedSoftmax,
    MaskedSoftmaxConfig,
    MaskedSoftmaxKernel,
)
from scaling.core.nn.attention import cumulative_seq_lengths_to_dense_attention_mask


@pytest.mark.masked_softmax
@pytest.mark.skipif((not torch.cuda.is_available()), reason="no cuda available")
@pytest.mark.parametrize("batch_size", [2, 4, 6])
@pytest.mark.parametrize("n_heads", [4, 12, 16])
@pytest.mark.parametrize("seq_len", [128 * i for i in range(1, 17)])
@pytest.mark.parametrize("scale", [1.0, 2.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_masked_softmax(batch_size, n_heads, seq_len, scale, dtype):
    instantiated_implementations = list()
    for masked_softmax_kernel in MaskedSoftmaxKernel:
        if masked_softmax_kernel == MaskedSoftmaxKernel.FLASH_ATTENTION:
            # skip test for flash attention as it doesn't use the forward pass of the masked softmax class
            continue

        # Instantiate config to instantiate to be benchmarked module
        config = MaskedSoftmaxConfig(kernel=masked_softmax_kernel, softmax_in_fp32=False, scale=scale)

        # instantiate module dependent on config (e.g. with provider set)
        masked_softmax = MaskedSoftmax(config=config)

        instantiated_implementations.append((masked_softmax_kernel, masked_softmax))

    # seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len, device="cuda", dtype=dtype)
    attention_mask = cumulative_seq_lengths_to_dense_attention_mask(
        torch.tensor([0, seq_len], device="cuda"), seq_length_per_batch_item=seq_len, causal=True
    )

    assert len(instantiated_implementations) > 0, "no masked softmax implemented"

    # assert that all implementations yield the same result
    ground_truth_kernel = instantiated_implementations[0][0]
    ground_truth = instantiated_implementations[0][1](attention_scores, attention_mask)

    for i in range(1, len(instantiated_implementations)):
        kernel, implementation = instantiated_implementations[i]
        compare = implementation(attention_scores, attention_mask)
        mean_delta = (ground_truth - compare).abs().mean().item()
        # TODO this delta is quite high. Why? Do we really get as big numeric differences?
        assert mean_delta < 0.000004, (
            f"masked softmax implementations for {ground_truth_kernel} "
            f"and {kernel} yield different results with max delta of {mean_delta}"
        )
