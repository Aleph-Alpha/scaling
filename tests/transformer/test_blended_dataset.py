import os
import shutil

import pytest

from scaling.core import BlendedDatasetConfig
from scaling.transformer.data.text_dataset import (
    TextBlendedDataset,
    TextDataset,
)


def purge(data_prefix):
    try:
        os.unlink(f"{data_prefix}.bin")
        os.unlink(f"{data_prefix}.idx")
    except Exception:
        ...


def create_data_prefix(name):
    data_prefix = f"tests/transformer/files/generated/{name}"
    purge(data_prefix)
    return data_prefix


# pytest tests/test_data/test_blended_dataset.py::test_blended_dataset_multiple -s
@pytest.mark.transformer
@pytest.mark.cpu
def test_blended_dataset_multiple():
    if os.path.exists("tests/transformer/files/generated"):
        shutil.rmtree("tests/transformer/files/generated")

    data_file_jsonl_1 = "tests/transformer/files/data/small1.jsonl"
    data_file_jsonl_2 = "tests/transformer/files/data/small2.jsonl"
    data_file_jsonl_3 = "tests/transformer/files/data/small3.jsonl"

    data_prefix_1 = create_data_prefix("small1")
    data_prefix_2 = create_data_prefix("small2")
    data_prefix_3 = create_data_prefix("small3")

    blended_config = {
        "cache_directory": "tests/transformer/files/tmp/generated",
    }
    config = BlendedDatasetConfig(**blended_config)

    # test creation of dataset
    TextDataset.jsonl_to_memory_map(data_file_jsonl_1, data_prefix_1)
    TextDataset.jsonl_to_memory_map(data_file_jsonl_2, data_prefix_2)
    TextDataset.jsonl_to_memory_map(data_file_jsonl_3, data_prefix_3)

    sequence_length = 5
    seed = 42
    dataset_1 = TextDataset(data_prefix=data_prefix_1, sequence_length=sequence_length, seed=seed)
    dataset_2 = TextDataset(data_prefix=data_prefix_2, sequence_length=sequence_length, seed=seed)
    dataset_3 = TextDataset(data_prefix=data_prefix_3, sequence_length=sequence_length, seed=seed)
    blended_dataset = TextBlendedDataset(config=config, datasets=[dataset_1, dataset_2, dataset_3], seed=seed)

    assert len(blended_dataset) == len(dataset_1) + len(dataset_2) + len(dataset_3)

    _test = blended_dataset[0]
