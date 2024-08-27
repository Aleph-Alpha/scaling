from typing import List, Optional

import torch

from scaling.core import (
    BaseDataset,
    BaseDatasetBatch,
    BaseDatasetItem,
    Topology,
    broadcast_data,
)


class MinimalDatasetItem(BaseDatasetItem):
    input: torch.Tensor
    target: torch.Tensor

    def __init__(self, index: int):
        super().__init__()
        self.input = torch.tensor(
            [0, 1], dtype=torch.long
        )  # hardcoded tensors that are in range of the 2 rows minimal embedding matrix
        self.target = torch.tensor([2 * index + 1], dtype=torch.long)


class MinimalBatch(BaseDatasetBatch):
    inputs: torch.Tensor
    targets: torch.Tensor

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def only_inputs(self):
        """
        function removing all properties from batch that are not inputs (i.e. targets)
        this may be useful to reduce memory load
        """
        return self

    def only_targets(self):
        """
        function removing all properties from batch that are not targets (i.e. inputs)
        this may be useful to reduce memory load
        """
        return self


class MinimalDataset(BaseDataset[MinimalDatasetItem, MinimalBatch, MinimalBatch]):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def ident(self):
        return "MinimalDataset"

    def set_seed(self, seed: int, shuffle: bool = True):
        # return an empty dummy indexed dataset
        return

    def __len__(self):
        # added a number of more samples here
        # to don't get into weird data issues for test_training_resume_with_different_layout
        return 1000

    def __getitem__(self, index) -> MinimalDatasetItem:
        return MinimalDatasetItem(index=index)

    def collate(self, batch: List[MinimalDatasetItem]) -> MinimalBatch:
        """
        Used to collate lists of samples into batches
        The default implementation returns a BaseDataBatch NamedTuple
        with the same attributes as the first element of the batch
        """
        # assume that all batch items have the same data type
        # get list of fields with their type
        inputs = torch.stack(
            [batch_item.input for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work
        targets = torch.stack(
            [batch_item.target for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work

        return MinimalBatch(inputs=inputs, targets=targets)

    @staticmethod
    def sync_batch_to_model_parallel(topology: Topology, batch: Optional[MinimalBatch]) -> MinimalBatch:
        if topology.model_parallel_rank == 0:
            assert batch is not None
            tensors: List[Optional[torch.Tensor]] = [batch.inputs, batch.targets]
        else:
            assert batch is None
            tensors = [None, None]

        broadcast_tensors = broadcast_data(tensors=tensors, dtype=torch.long, topology=topology)  # type: ignore

        return MinimalBatch(inputs=broadcast_tensors[0], targets=broadcast_tensors[1])
