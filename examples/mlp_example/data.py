from pathlib import Path
from typing import Any

import torch
import torchvision
from torchvision import transforms

from scaling.core import BaseDataset, BaseDatasetBatch, BaseDatasetItem, broadcast_data
from scaling.core.topology import Topology


class MNISTDatasetItem(BaseDatasetItem):
    def __init__(self, input_: Any, target: Any) -> None:
        self.input = torch.tensor(input_, dtype=torch.float16)
        self.target = torch.tensor(target, dtype=torch.float16)


class MNISTDatasetBatch(BaseDatasetBatch):
    def __init__(
        self,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ):
        self.inputs = inputs
        self.targets = targets

    def only_inputs(self) -> "MNISTDatasetBatch":
        return MNISTDatasetBatch(inputs=self.inputs)

    def only_targets(self) -> "MNISTDatasetBatch":
        return MNISTDatasetBatch(targets=self.targets)


class MNISTDataset(BaseDataset[MNISTDatasetItem, MNISTDatasetBatch, MNISTDatasetBatch]):
    def __init__(self, root: Path = Path("./.data"), train: bool = True):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> MNISTDatasetItem:
        return MNISTDatasetItem(
            input_=self.dataset[index][0],
            target=self.dataset[index][1],
        )

    def ident(self) -> str:
        return "MNIST"

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        return

    def collate(self, batch: list[MNISTDatasetItem]) -> MNISTDatasetBatch:
        inputs = torch.stack([batch_item.input for batch_item in batch])
        targets = torch.stack([batch_item.target for batch_item in batch])
        return MNISTDatasetBatch(inputs=inputs, targets=targets)

    @staticmethod
    def sync_batch_to_model_parallel(
        topology: Topology,
        batch: MNISTDatasetBatch | None,
    ) -> MNISTDatasetBatch:
        if topology.model_parallel_rank == 0:
            assert batch is not None
            tensors: list[torch.Tensor | None] = [batch.inputs, batch.targets]
        else:
            assert batch is None
            tensors = [None, None]

        broadcast_tensors = broadcast_data(tensors=tensors, dtype=torch.float16, topology=topology)

        return MNISTDatasetBatch(
            inputs=broadcast_tensors[0],
            targets=broadcast_tensors[1],
        )
