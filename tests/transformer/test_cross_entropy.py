import pytest
import torch

from scaling.transformer.context.config import CrossEntropyLossFunctionConfig, UMuPConfig
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.model.layers.base import TransformerLayerIO
from scaling.transformer.model.losses.cross_entropy import CrossEntropyLoss


def is_close(x: float, x_ref: float, rtol: float = 5.0e-2) -> bool:
    return abs(x - x_ref) / x_ref < rtol


def has_scale(tensor: torch.Tensor, scale: float, rtol: float = 5.0e-2) -> bool:
    return is_close(tensor.std().item(), scale, rtol=rtol)


@pytest.mark.parametrize("mult", [1.0, 2.0])
def test_umup_cross_entropy(mult: float):
    batch_size = 16
    vocab_size = 4096
    sequence_length = 1024

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    loss_config = CrossEntropyLossFunctionConfig()
    umup_config = UMuPConfig(loss_mult=mult, enable=True)
    cross_entropy = CrossEntropyLoss(loss_config=loss_config, umup_config=umup_config)

    activations = torch.randn(batch_size, sequence_length, vocab_size).cuda()
    activations.requires_grad = True

    target_token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length)).cuda()
    loss_weights = torch.ones_like(target_token_ids)
    target_batch = TextDatasetBatch(target_token_ids=target_token_ids, loss_weights=loss_weights)
    # cum seq len and pos ids not relevant here
    loss_input = TransformerLayerIO(
        activations=activations, cumulative_seq_lengths_padded=torch.zeros((0,)), position_ids=torch.zeros((0,))
    )

    loss = cross_entropy(loss_input, target_batch)[0]
    loss.backward()

    assert activations.grad is not None
    assert has_scale(activations.grad, 1.0), activations.grad.std()

    if mult == 1.0:
        cross_entropy_standard = CrossEntropyLoss(CrossEntropyLossFunctionConfig(), UMuPConfig())
        loss_standard = cross_entropy_standard(loss_input, target_batch)[0]
        assert is_close(loss, loss_standard)
