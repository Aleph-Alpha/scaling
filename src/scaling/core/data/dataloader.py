# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Generator, Optional

import torch

from scaling.core.data import BaseDataset
from scaling.core.logging import logger

from ..topology import Topology


class RandomSampler:
    """
    Class with iterator that returns a list of indices for a micro batch.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        seed: int,
        consumed_samples: int,
        topology: Topology,
        shuffle: bool = True,
    ) -> None:
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.topology = topology
        self.shuffle = shuffle

        # derive parameters
        self.total_samples = len(dataset)
        self.total_micro_batches = len(self.dataset) // self.topology.config.micro_batch_size
        self.total_micro_batches_per_data_parallel = self.total_micro_batches // self.topology.config.data_parallel_size
        self.usable_total_samples = (
            self.total_micro_batches_per_data_parallel
            * self.topology.config.micro_batch_size
            * self.topology.config.data_parallel_size
        )
        assert self.usable_total_samples > 0, (
            "not usable samples; "
            "this means that the dataset is too small for the provided data parallel size and micro batch size"
        )

    def __len__(self) -> int:
        return self.total_micro_batches

    def __iter__(self) -> Generator[Any, None, None]:
        epoch = self.consumed_samples // self.usable_total_samples
        consumed_samples_in_current_epoch = self.consumed_samples % self.usable_total_samples
        remaining_samples_in_current_epoch = self.usable_total_samples - consumed_samples_in_current_epoch

        logger.info(f"creating new dataset shuffle index for epoch {epoch}")
        logger.info(f"total_samples {self.total_samples}")
        logger.info(f"micro_batch_size {self.topology.config.micro_batch_size}")
        logger.info(f"usable_total_samples {self.usable_total_samples}")
        logger.info(f"consumed_samples_in_current_epoch {consumed_samples_in_current_epoch}")
        logger.info(f"remaining_samples_in_current_epoch {remaining_samples_in_current_epoch}")

        self.dataset.set_seed(seed=self.seed + epoch, shuffle=self.shuffle)

        idx_range_tensor = (
            (
                torch.arange(
                    0,
                    (remaining_samples_in_current_epoch // self.topology.config.data_parallel_size),
                    dtype=torch.long,
                )
                * self.topology.config.data_parallel_size
            )
            + self.topology.data_parallel_rank
            + consumed_samples_in_current_epoch
        )

        idx_range = idx_range_tensor.tolist()
        assert self.topology.config.micro_batch_size is not None
        assert (
            len(idx_range) % self.topology.config.micro_batch_size == 0
        ), "dataset index count is not a multiple of micro batch size"

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.topology.config.micro_batch_size:
                self.consumed_samples += self.topology.config.micro_batch_size * self.topology.config.data_parallel_size
                yield batch
                batch = []


class DataLoader(torch.utils.data.DataLoader):
    """
    Generic base class to iterate over any given dataset which implements BaseDataset.

    The data loader
        - is instantiated from a seed, the number of consumed samples, a micro batch size and a dataset
        - implements an infinite iterator over the dataset
    """

    def __init__(
        self,
        seed: int,
        consumed_samples: int,
        dataset: BaseDataset,
        topology: Topology,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        """
        seed (`int`)
            seed used to shuffle the dataset
        consumed_samples (`int`)
            number of samples already consumed during training from the dataset

        dataset (`BaseDataset`)
            dataset which implements the BaseDataset interface
        """
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.dataset = dataset
        self.topology = topology

        assert len(self.dataset) >= self.topology.config.micro_batch_size, (
            f"cannot instantiate data loader with micro_batch_size {self.topology.config.micro_batch_size} "
            f"because dataset has only length {len(self.dataset)}"
        )

        batch_sampler = RandomSampler(
            dataset=self.dataset,
            seed=self.seed,
            consumed_samples=self.consumed_samples,
            topology=self.topology,
            shuffle=shuffle,
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.dataset.collate,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        self.iterator = self._iterate()

    def _iterate(self) -> Generator[Any, None, None]:
        while True:
            for item in self.dataloader:
                yield item

    def __next__(self) -> Any:
        return next(self.iterator)
