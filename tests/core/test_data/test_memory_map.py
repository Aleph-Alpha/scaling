from pathlib import Path

import numpy as np
import pytest

from scaling.core import MemoryMapDataset, MemoryMapDatasetBuilder


def test_memory_map(tmp_path: Path):
    """
    tests the creation and read of a memory map dataset
    """

    prefix_path = tmp_path / "data_set_prefix"

    data_items = [
        [1, 2, 3, 4, 5],
        [1, 2, 5],
        [45, 1, 20, 303, 30203],
    ]

    # instantiate a builder and write data
    builder = MemoryMapDatasetBuilder(prefix_path=prefix_path)

    for data_item in data_items:
        builder.add(np_array=np.array(data_item))
    builder.finalize()

    # make sure an error is raised if the dataset already exist
    with pytest.raises(AssertionError):
        builder = MemoryMapDatasetBuilder(prefix_path=prefix_path)

    # load the dataset
    dataset = MemoryMapDataset(prefix_path=prefix_path)
    assert len(dataset) == len(data_items)

    # compare all data items to ground truth
    for data_item_dataset, data_item_truth in zip(dataset, data_items):
        assert np.equal(data_item_dataset, np.array(data_item_truth)).all()
