from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

import torch

from scaling.core.data.base_layer_io import BaseLayerIO
from scaling.core.topology import Topology

TBaseDatasetBatch = TypeVar("TBaseDatasetBatch", bound="BaseDatasetBatch")


class BaseDatasetItem:
    """
    Base class for dataset items, providing an interface that all dataset items have to implement
    """


class BaseDatasetBatch(BaseLayerIO):
    """
    Base class for dataset items, providing an interface that all dataset items have to implement
    """

    @abstractmethod
    def only_inputs(self: TBaseDatasetBatch) -> TBaseDatasetBatch:
        """
        function removing all properties from batch that are not inputs (i.e. targets)
        this may be useful to reduce memory load
        """
        return self

    @abstractmethod
    def only_targets(self: TBaseDatasetBatch) -> TBaseDatasetBatch:
        """
        function removing all properties from batch that are not targets (i.e. inputs)
        this may be useful to reduce memory load
        """
        return self


BaseDatasetItemGeneric = TypeVar("BaseDatasetItemGeneric", bound=BaseDatasetItem)
BaseDatasetBatchBeforeSyncGeneric = TypeVar("BaseDatasetBatchBeforeSyncGeneric", bound=BaseDatasetBatch)
BaseDatasetBatchGeneric = TypeVar("BaseDatasetBatchGeneric", bound=BaseDatasetBatch)


class BaseDataset(
    torch.utils.data.Dataset,
    Generic[
        BaseDatasetItemGeneric,
        BaseDatasetBatchBeforeSyncGeneric,
        BaseDatasetBatchGeneric,
    ],
):
    """
    Torch base dataset class expected to be inherited by all datasets.
    Returns a BaseDatasetItem for each index.
    """

    def __init__(
        self,
        seed: int,
        data_prefix: Path = Path(""),
        data_index_prefix: Path | None = None,
        shuffle: bool = True,
    ) -> None:
        """
        seed (`int`)
            seed used to shuffle the dataset

        data_prefix (`str`)
            prefix of the data files

        shuffle (`bool`)
            whether to shuffle the dataset
        """
        # shuffling
        self.seed: Optional[int] = None
        self.data_prefix = data_prefix
        self.data_index_prefix = data_index_prefix if data_index_prefix is not None else data_prefix
        self.set_seed(seed=seed, shuffle=shuffle)

    @abstractmethod
    def ident(self) -> str:
        # implement in child class
        raise NotImplementedError

    def get_data_index_cache_filename_stem(self, seed: int) -> str:
        cache_file = str(self.data_index_prefix) + f"_index_cache_decoder_dataset_seed_{seed}"
        return cache_file

    def get_data_index_cache_filename_bin(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".bin"

    def get_data_index_cache_filename_idx(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".idx"

    def get_data_index_cache_filename_done(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".done"

    def get_data_index_cache_filename_meta(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".meta.json"

    @abstractmethod
    def __len__(self) -> int:
        # implement in child class
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> BaseDatasetItemGeneric:
        """
        Returns a BaseDatasetItem for each index
        """
        # implement in child class
        raise NotImplementedError

    @abstractmethod
    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        """
        Sets the seed for shuffling the dataset
        """
        # implement in child class
        raise NotImplementedError

    @abstractmethod
    def collate(self, batch: list[BaseDatasetItemGeneric]) -> BaseDatasetBatchBeforeSyncGeneric:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sync_batch_to_model_parallel(
        topology: Topology, batch: Optional[BaseDatasetBatchBeforeSyncGeneric]
    ) -> BaseDatasetBatchGeneric:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
