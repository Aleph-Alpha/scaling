from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pytest
import torch

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    BaseDatasetBatch,
    BaseDatasetItem,
    BlendedDatasetConfig,
    DataLoader,
    Topology,
    TopologyConfig,
)


# utils
def isinstance_namedtuple(obj) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def get_topology(data_parallel_size: int, micro_batch_size: int, global_rank: int):
    return Topology(
        config=TopologyConfig(  # type: ignore[call-arg]
            global_rank=global_rank,
            model_parallel_size=1,
            data_parallel_size=data_parallel_size,
            pipe_parallel_size=1,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
        )
    )


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4, 5, 6])
@pytest.mark.parametrize("data_parallel_size", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dataset_size", [10, 17, 20])
@pytest.mark.parametrize("blended_config", [None, {"weight_by_num_documents": True}])
def test_dataloader(
    tmp_path: Path,
    micro_batch_size: int,
    data_parallel_size: int,
    dataset_size: int,
    blended_config: Optional[Dict],
):
    """
    verifies instantiation of a data loader and that the output format is as expected
    """
    if dataset_size < micro_batch_size * data_parallel_size:
        pytest.skip("not enough data for provided micro_batch_size and data_parallel_size")

    class MockDatasetItem(BaseDatasetItem):
        token_ids: torch.Tensor

        def __init__(self, index, multiply: int, add: int):
            super().__init__()
            self.token_ids = torch.tensor([index * multiply + add])

    class MockDatasetBatch(BaseDatasetBatch):
        token_ids: torch.Tensor

        def __init__(self, token_ids):
            super().__init__()
            self.token_ids = token_ids

        def only_inputs(self):
            return self

        def only_targets(self):
            return self

    class MockDataset(BaseDataset[MockDatasetItem, MockDatasetBatch, MockDatasetBatch]):
        def __init__(self, seed, multiply: int = 1, add: int = 0):
            super().__init__(seed=seed)
            self.multiply = multiply
            self.add = add

        def set_seed(self, seed: int, shuffle: bool = True):
            # return an empty dummy indexed dataset
            return

        def ident(self):
            return "mock"

        def __len__(self):
            return dataset_size

        def __getitem__(self, index) -> MockDatasetItem:
            return MockDatasetItem(index=index, multiply=self.multiply, add=self.add)

        def collate(self, batch: List[MockDatasetItem]) -> MockDatasetBatch:
            """
            Used to collate lists of samples into batches
            The default implementation returns a BaseDataBatch NamedTuple
            with the same attributes as the first element of the batch
            """
            # assume that all batch items have the same data type
            # get list of fields with their type
            return MockDatasetBatch(
                token_ids=torch.stack(
                    [batch_item.token_ids for batch_item in batch]
                )  # don't move to cuda, otherwise background data loader processes will not work
            )

        @staticmethod
        def sync_batch_to_model_parallel(topology: Topology, batch: Optional[MockDatasetBatch]) -> MockDatasetBatch:
            # return empty mock dataset that is not used for this test
            return MockDatasetBatch(token_ids=torch.empty())

    class MockBlendedDataset(BaseBlendedDataset[MockDatasetItem, MockDatasetBatch, MockDatasetBatch, MockDataset]):
        pass

    def get_all_data(consumed_samples: int):
        # initialize dataset
        dataset: Union[MockDataset, MockBlendedDataset]
        if blended_config is None:
            dataset = MockDataset(seed=42)
        else:
            dataset = MockBlendedDataset(
                seed=42,
                config=BlendedDatasetConfig(**blended_config, cache_directory=tmp_path),
                datasets=[
                    MockDataset(seed=42, multiply=data_parallel_size, add=dp) for dp in range(data_parallel_size)
                ],
            )

        # initialize data loader class
        data_loader_by_dp_rank: Dict[int, DataLoader] = dict()
        for data_parallel_rank in range(data_parallel_size):
            data_loader_by_dp_rank[data_parallel_rank] = DataLoader(
                seed=42,
                consumed_samples=0,
                dataset=dataset,
                topology=get_topology(
                    data_parallel_size=data_parallel_size,
                    micro_batch_size=micro_batch_size,
                    global_rank=data_parallel_rank,
                ),
            )

        # Make sure we can iterate on more than the dataset length
        all_batch_tensors = list()
        for _ in range(0, len(data_loader_by_dp_rank[0].dataset) + 2):  # type: ignore
            all_micro_batch_tensors = list()
            for data_parallel_rank in range(data_parallel_size):
                micro_batch = next(data_loader_by_dp_rank[data_parallel_rank])
                all_micro_batch_tensors.append(micro_batch.token_ids)
                assert isinstance(
                    micro_batch, BaseDatasetBatch
                ), f"{micro_batch} with default collate is not a BaseDatasetBatch"
                for key, value in micro_batch.__dict__.items():
                    assert (
                        value.shape[0] == micro_batch_size
                    ), f"batched {key} do not have shape of length micro_batch_size"

            # collect full batch
            full_batch = torch.stack(all_micro_batch_tensors).transpose(1, 0).flatten()
            assert full_batch.numel() == micro_batch_size * data_parallel_size
            all_batch_tensors.append(full_batch)

        all_data = torch.stack(all_batch_tensors).flatten().cpu().tolist()
        return all_data

    is_shuffled = blended_config is not None and blended_config.get("shuffle_dataset_indices")
    # make sure that data is properly spread across data parallel
    all_data = get_all_data(consumed_samples=0)
    for i in range(len(all_data) - 1):
        if not is_shuffled:
            if all_data[i + 1] > 0:
                assert (
                    all_data[i] + 1 == all_data[i + 1]
                ), f"data is not interleaved on data parallel; all data {all_data}"

    # test resume from within one epoch
    consumed_sample_within_epoch = dataset_size // 2
    all_data_within_epoch_resume = get_all_data(consumed_samples=consumed_sample_within_epoch)
    baseline = all_data[consumed_sample_within_epoch:]
    all_data_within_epoch_resume_applicable = all_data_within_epoch_resume[-len(baseline) :]
    if not is_shuffled:
        assert (
            baseline == all_data_within_epoch_resume_applicable
        ), "dataset did not resume correctly with checkpoint within first epoch"

    # test resume from more than one epoch
    consumed_sample_more_than_one_epoch = int(dataset_size * 1.2)
    all_data_within_epoch_resume = get_all_data(consumed_samples=consumed_sample_more_than_one_epoch)
    baseline = all_data[consumed_sample_within_epoch:]
    consumed_sample_more_than_one_epoch_applicable = all_data_within_epoch_resume[-len(baseline) :]
    if not is_shuffled:
        assert (
            baseline == consumed_sample_more_than_one_epoch_applicable
        ), "dataset did not resume correctly with checkpoint after first epoch"


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4, 6])
def test_dataloader_with_custom_collate_fn(micro_batch_size):
    """
    Tests a dataloader with a collate function with custom `torch.Tensor` return type
    """

    class MockDatasetItem(BaseDatasetItem):
        def __init__(self):
            super().__init__()
            self.token_ids = torch.tensor([1, 2, 3, 4, 5])

    class MockDatasetBatch(BaseDatasetBatch):
        def __init__(self, token_ids: torch.Tensor):
            super().__init__()
            self.token_ids = token_ids

        def only_inputs(self):
            return self

        def only_targets(self):
            return self

    class MockDataset(BaseDataset[MockDatasetItem, MockDatasetBatch, MockDatasetBatch]):
        def __init__(self, seed):
            super().__init__(seed=seed)

        def set_seed(self, seed: int, shuffle: bool = True):
            # return an empty dummy indexed dataset
            return

        def __len__(self):
            return 10

        def __getitem__(self, index) -> MockDatasetItem:
            return MockDatasetItem()

        def collate(self, batch: Iterable[MockDatasetItem]) -> MockDatasetBatch:
            return MockDatasetBatch(torch.stack([item.token_ids for item in batch]))

    # initialize dataset
    dataset = MockDataset(seed=42)

    # initialize data loader class
    dataloader = DataLoader(
        seed=42,
        consumed_samples=0,
        dataset=dataset,
        topology=get_topology(data_parallel_size=1, micro_batch_size=micro_batch_size, global_rank=0),
    )

    # Make sure we can iterate on more than the dataset length
    for _ in range(0, len(dataloader.dataset) + 2):
        batch = next(dataloader)
        assert isinstance(
            batch, MockDatasetBatch
        ), f"{batch} with custom collate and custom data_item_batch_to_io is not a MockDatasetBatch"
        assert (
            batch.token_ids.shape[0] == micro_batch_size
        ), "batched tensor does not have shape of length micro_batch_size"


@pytest.mark.parametrize("n_datasets", [2, 3])
def test_dataloader_blended_shuffle(
    tmp_path: Path,
    n_datasets: int,
):
    """
    verifies instantiation of a blended dataloader with shuffling, and that indices are shuffled in ways we expect.
    not done for single dataset case, since we only care about testing blended dataset indices
    """

    class MockDatasetItem(BaseDatasetItem):
        def __init__(self):
            super().__init__()
            self.token_ids = torch.tensor([1, 2, 3, 4, 5])

    class MockDatasetBatch(BaseDatasetBatch):
        def __init__(self, token_ids: torch.Tensor):
            super().__init__()
            self.token_ids = token_ids

        def only_inputs(self):
            return self

        def only_targets(self):
            return self

    class MockDataset(BaseDataset[MockDatasetItem, MockDatasetBatch, MockDatasetBatch]):
        def __init__(self, seed):
            super().__init__(seed=seed)

        def ident(self):
            return "mock"

        def set_seed(self, seed: int, shuffle: bool = True):
            # return an empty dummy indexed dataset
            return

        def __len__(self):
            return 10

        def __getitem__(self, index) -> MockDatasetItem:
            return MockDatasetItem()

        def collate(self, batch: Iterable[MockDatasetItem]) -> MockDatasetBatch:
            return MockDatasetBatch(torch.stack([item.token_ids for item in batch]))

        @staticmethod
        def sync_batch_to_model_parallel(topology: Topology, batch: Optional[MockDatasetBatch]) -> MockDatasetBatch:
            # return empty mock dataset that is not used for this test
            return MockDatasetBatch(token_ids=torch.empty())

    class MockBlendedDataset(BaseBlendedDataset[MockDatasetItem, MockDatasetBatch, MockDatasetBatch, MockDataset]):
        pass

    # initialize dataset
    datasets = [MockDataset(seed=i) for i in range(n_datasets)]

    # Get all_data without shuffling of blended dataset indices
    blended_config_without_shuffle = {"shuffle_dataset_indices": False}
    blended_config_with_shuffle = {"shuffle_dataset_indices": True}

    unshuffled_blended_dataset = MockBlendedDataset(
        seed=42,
        config=BlendedDatasetConfig(**blended_config_without_shuffle, cache_directory=tmp_path),
        datasets=datasets,
    )

    shuffled_blended_dataset = MockBlendedDataset(
        seed=42,
        config=BlendedDatasetConfig(**blended_config_with_shuffle, cache_directory=tmp_path),
        datasets=datasets,
    )

    # simulate initialization of blended dataset for 2 different epochs
    shuffled_blended_dataset_first_epoch = deepcopy(shuffled_blended_dataset)
    shuffled_blended_dataset_first_epoch.set_seed(42)
    shuffled_blended_dataset_second_epoch = deepcopy(shuffled_blended_dataset_first_epoch)
    shuffled_blended_dataset_second_epoch.set_seed(1)

    # dataset_indices only exists if more than one dataset is loaded together
    # test that shuffling did mix the indices as expected
    assert shuffled_blended_dataset.random_index is not None
    assert unshuffled_blended_dataset.random_index is not None
    assert not np.array_equal(
        shuffled_blended_dataset.random_index, unshuffled_blended_dataset.random_index
    ), "shuffling didn't occur as expected"

    # test that shuffling at different epochs results in different ordering of indices
    assert shuffled_blended_dataset_first_epoch.random_index is not None
    assert shuffled_blended_dataset_second_epoch.random_index is not None
    assert not np.array_equal(
        shuffled_blended_dataset_first_epoch.random_index,
        shuffled_blended_dataset_second_epoch.random_index,
    ), "shuffling at different epochs didn't result in different indices"

    # test shuffling determinism
    shuffled_blended_dataset_second_epoch.set_seed(42)  # same seed as first epoch blended dataset
    assert np.array_equal(
        shuffled_blended_dataset_first_epoch.random_index,
        shuffled_blended_dataset_second_epoch.random_index,
    ), "the same seed does not return the same dataset"
