from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import torch

from ..topology import Topology
from .base_layer_io import BaseLayerIO

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

    def __init__(self, seed: int, shuffle: bool = True) -> None:
        """
        seed (`int`)
            seed used to shuffle the dataset

        config (`Optional[BaseConfig]`)
            dataset config which need to be implemented in child classes
        """
        # shuffling
        self.seed: Optional[int] = None
        self.set_seed(seed=seed, shuffle=shuffle)

    @abstractmethod
    def ident(self) -> str:
        # implement in child class
        raise NotImplementedError

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
