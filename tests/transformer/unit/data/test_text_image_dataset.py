import uuid
from pathlib import Path

import pytest
import torch

from scaling.core.data.blended_dataset_config import BlendedDatasetConfig
from scaling.transformer.data.text_image_dataset import TextImageBlendedDataset, TextImageDataset


@pytest.fixture
def dataset():
    return TextImageDataset(
        data_prefix=Path("tests/transformer/files/dataset/text_image_data"),
        data_index_prefix=Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}"),
        sequence_length=512,
        softprompt_n_tokens=0,
    )


@pytest.fixture
def dataset_list():
    return [
        TextImageDataset(
            data_prefix=Path("tests/transformer/files/dataset/text_image_data"),
            data_index_prefix=Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}"),
            sequence_length=512,
            softprompt_n_tokens=0,
        )
        for _ in range(3)
    ]


@pytest.mark.unit
def test_ident(dataset: TextImageDataset):
    ident = dataset.ident()
    assert ident.endswith(f"-seq-{dataset.sequence_length}")
    assert ident.split("-")[0]


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11])
def test_get_item(dataset: TextImageDataset, index: int):
    assert dataset.data_item_index is not None
    mapped_indices = dataset.data_item_index[index]
    raw_data = dataset.memory_map[mapped_indices[0]]
    data = dataset[index]

    softprompt_n_tokens = dataset.softprompt_n_tokens if hasattr(dataset, "softprompt_n_tokens") else 0
    sequence_length = (
        dataset.sequence_length if hasattr(dataset, "sequence_length") else len(raw_data["input_token_list"])
    )

    adjusted_input_token_list = ([0] * softprompt_n_tokens + raw_data["input_token_list"])[:sequence_length]
    adjusted_target_token_list = ([0] * softprompt_n_tokens + raw_data["target_token_list"])[:sequence_length]
    adjusted_loss_mask_list = ([0] * softprompt_n_tokens + raw_data["loss_mask_list"])[:sequence_length]

    padding_size = sequence_length - len(adjusted_input_token_list)
    adjusted_input_token_list += [dataset.padding_token_id] * padding_size
    adjusted_target_token_list += [dataset.padding_token_id] * padding_size
    adjusted_loss_mask_list += [0] * padding_size

    expected_input_token_ids = torch.tensor(adjusted_input_token_list, dtype=torch.long)
    expected_target_token_ids = torch.tensor(adjusted_target_token_list, dtype=torch.long)
    expected_loss_weights = torch.tensor(adjusted_loss_mask_list, dtype=torch.float32)

    assert torch.equal(data.input_token_ids, expected_input_token_ids)
    assert torch.equal(data.target_token_ids, expected_target_token_ids)
    assert torch.equal(data.loss_weights, expected_loss_weights)
    expected_position_ids = torch.arange(0, sequence_length)
    assert torch.equal(data.position_ids, expected_position_ids)


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 4, 8, 11])
def test_seed(dataset: TextImageDataset, index: int):
    dataset.set_seed(seed=42)
    first_item = dataset[index]

    dataset.set_seed(seed=42)
    second_item = dataset[index]

    assert torch.equal(first_item.input_token_ids, second_item.input_token_ids)
    assert torch.equal(first_item.target_token_ids, second_item.target_token_ids)

    dataset.set_seed(seed=24)
    fourth_item = dataset[index]

    assert not torch.equal(first_item.input_token_ids, fourth_item.input_token_ids)
    assert not torch.equal(first_item.target_token_ids, fourth_item.target_token_ids)


@pytest.mark.unit
@pytest.mark.parametrize("batch_indices", [[0, 1, 2], [4, 5, 6], [7, 8, 9], [10, 11]])
def test_collate(dataset: TextImageDataset, batch_indices: list[int]):
    batch_list = [dataset[i] for i in batch_indices]
    batch_collated = dataset.collate(batch_list)

    image_count = 0
    for i, batch_item in enumerate(batch_list):
        assert batch_collated.input_token_ids is not None
        assert batch_collated.target_token_ids is not None
        assert batch_collated.position_ids is not None
        assert batch_collated.loss_weights is not None
        assert torch.equal(batch_collated.input_token_ids[i], batch_item.input_token_ids)
        assert torch.equal(batch_collated.target_token_ids[i], batch_item.target_token_ids)
        assert torch.equal(batch_collated.position_ids[i], batch_item.position_ids)
        assert torch.equal(batch_collated.loss_weights[i], batch_item.loss_weights)

        if batch_item.input_images:
            assert batch_collated.input_images is not None
            assert torch.equal(batch_collated.input_images[image_count], batch_item.input_images[0])
            image_count += 1


@pytest.mark.unit
def test_blended_dataset(dataset_list: list[TextImageDataset]):
    blended_config = {
        "cache_directory": str(Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")),
    }

    config = BlendedDatasetConfig(**blended_config)
    blended_dataset = TextImageBlendedDataset(config=config, datasets=dataset_list, seed=42)

    assert len(blended_dataset) == sum(len(dataset) for dataset in dataset_list)
    test_item = blended_dataset[0]
    print(test_item)
