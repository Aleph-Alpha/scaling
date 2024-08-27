import contextlib
from datetime import timedelta
from typing import Any, Optional, TypedDict

import torch
from torch.distributed import ProcessGroup

from scaling.core.logging import logger

from .rng_tracker import CudaRNGStateTracker, RngTrackerState
from .topology_config import TopologyConfig


class TopologyState(TypedDict):
    model_parallel_constant_rng: RngTrackerState
    model_parallel_size: int
    pipe_parallel_size: int


class Topology:
    """
    Layout of one full train or inference job on nodes (and gpus).

    This class implements functionality of Megatron's mpu.
    """

    def __init__(self, config: TopologyConfig):
        """
        config (`TopologyConfig`)
            Validated TopologyConfig
        """

        # Remember layout parameters
        self.config = config
        self.is_distributed_initialized = False

        # a world size is needed to initialize topology
        assert self.config.world_size is not None
        assert self.config.data_parallel_size is not None

        # Derive 3D parallelism layout
        # Make a 3D grid (pipeline, data and model parallelism) out of all ranks
        # The order of the dimensions implies the priority for being on the same node:
        # model parallel first, then data parallel, then pipe parallel
        self._layout_3D = torch.arange(self.config.world_size).reshape(
            self.config.pipe_parallel_size,
            self.config.data_parallel_size,
            self.config.model_parallel_size,
        )

        (
            self._pipe_parallel_rank,
            self._data_parallel_rank,
            self._model_parallel_rank,
        ) = (self._layout_3D == self.config.global_rank).nonzero()[0].tolist()

        # Record groups for current global_rank
        # All other groups and properties are the same for all ranks
        # The values will be filled in self.initialize_distributed()

        # all pipe stages with one constant model parallel and data parallel
        self._pipe_parallel_ranks: Optional[list[int]] = None
        self._pipe_parallel_group = None

        # all data stages with one constant model parallel and pipe parallel
        self._data_parallel_ranks: Optional[list[int]] = None
        self._data_parallel_group = None

        # all model stages with one constant data parallel and pipe parallel
        self._model_parallel_ranks: Optional[list[int]] = None
        self._model_parallel_group = None

        # current torch device
        self._device: Optional[torch.device] = None

        self._model_parallel_constant_rng: Optional[CudaRNGStateTracker] = None

    @property
    def has_model_parallel_constant_rng(self) -> bool:
        return self._model_parallel_constant_rng is not None

    @property
    def model_parallel_constant_rng(self) -> Any:
        if not self.has_model_parallel_constant_rng:
            return contextlib.nullcontext
        return self._model_parallel_constant_rng.fork  # type: ignore[union-attr]

    def state_dict(self) -> Optional[TopologyState]:
        if not self.has_model_parallel_constant_rng:
            return None
        return {
            "model_parallel_constant_rng": self._model_parallel_constant_rng.state_dict(),  # type: ignore[union-attr]
            "model_parallel_size": self.config.model_parallel_size,
            "pipe_parallel_size": self.config.pipe_parallel_size,
        }

    def load_state_dict(self, state_dict: Optional[dict[str, Any]]) -> None:
        if not self.has_model_parallel_constant_rng or state_dict is None:
            return

        # only load if layout did not change. otherwise we may get inconsistent states
        if (
            self.config.model_parallel_size == state_dict["model_parallel_size"]
            and self.config.pipe_parallel_size == state_dict["pipe_parallel_size"]
        ):
            assert self._model_parallel_constant_rng is not None
            self._model_parallel_constant_rng.load_state_dict(state_dict["model_parallel_constant_rng"])

    def initialize_device(self) -> None:
        # Manually set the device ids.
        assert self.config.local_slot is not None, "cannot initialize distributed without local_slot defined"
        assert torch.cuda.is_available(), "no cuda available"
        assert (
            self.config.local_slot < torch.cuda.device_count()
        ), f"cannot assign gpu {self.config.local_slot} for {torch.cuda.device_count()} available gpus"
        torch.cuda.set_device(self.config.local_slot)
        self._device = torch.device(self.config.local_slot)

    def initialize_distributed(
        self,
        master_addr: str,
        master_port: str,
        torch_distributed_timeout_minutes: int = 20,
        seed: int = 42,
    ) -> None:
        logger.info(
            f"Topology.initialize_distributed() using master {master_addr}:{master_port} "
            f"with world_size {self.config.world_size} for rank {self.config.global_rank}"
        )
        assert not self.is_distributed_initialized, "distributed is already initialized"
        assert not torch.distributed.is_initialized(), "torch distributed is initialized. This is unexpected."

        self.initialize_device()

        self._model_parallel_constant_rng = CudaRNGStateTracker(seed=seed + self.get_global_rank(model_parallel_rank=0))

        # topology can only be initialized of world_size is defined in the config
        assert self.config.world_size is not None, "cannot initialize distributed without world size"

        assert self.config.global_rank is not None, "cannot initialize distributed without global rank"

        # Call the init process
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.config.world_size,
            rank=self.config.global_rank,
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(minutes=torch_distributed_timeout_minutes),
        )

        # Record groups for current global_rank
        # All other groups and properties are the same for all ranks

        # Note on the below strange looking implementation
        # The following restrictions apply
        # - all ranks must call torch.distributed.new_group
        #   in the same order with the same ranks as argument (same group)
        # - A new group may not be empty
        # => we initialize a group in any case and just remember what we need
        #
        # Example going RIGHT, each line below is called in that order on the different ranks
        # Rank 0:               Rank 1:             Rank 2:             Rank 3:
        # new_group([0, 1])     new_group([0, 1])   new_group([0, 1])   new_group([0, 1])
        # store new group       store new group     pass                pass
        # new_group([2, 3])     new_group([2, 3])   new_group([2, 3])   new_group([2, 3])
        # pass                  pass                store new group     store new group
        #
        # Example going WRONG, each line below is called in that order on the different ranks
        # Rank 0:               Rank 1:             Rank 2:             Rank 3:
        # new_group([0, 1])     new_group([0, 1])   new_group([2, 3])   new_group([2, 3])
        # store new group       store new group     store new group     store new group
        #                                                               here conflicting groups are created

        # TODO illustrate
        # Build pipe parallel group
        # 	dp	0					dp	1

        #     model	0	1				model	0	1
        # pipe						pipe
        #     0	    X				0
        #     1	    X				1
        #     2	    X				2
        #     3	    X				3

        # group must be initialized for all ranks even if it is not on rank
        for pipe_parallel_ranks in self.all_pipe_parallel_groups:
            group = torch.distributed.new_group(pipe_parallel_ranks)
            if self.config.global_rank in pipe_parallel_ranks:
                self._pipe_parallel_ranks = pipe_parallel_ranks
                self._pipe_parallel_group = group

        # Build data parallel group
        for data_parallel_ranks in self.all_data_parallel_groups:
            group = torch.distributed.new_group(data_parallel_ranks)
            if self.config.global_rank in data_parallel_ranks:
                self._data_parallel_ranks = data_parallel_ranks
                self._data_parallel_group = group

        # Build model parallel group
        for model_parallel_ranks in self.all_model_parallel_groups:
            group = torch.distributed.new_group(model_parallel_ranks)
            if self.config.global_rank in model_parallel_ranks:
                self._model_parallel_ranks = model_parallel_ranks
                self._model_parallel_group = group

        self.is_distributed_initialized = True

    @property
    def pipe_parallel_indices(self) -> list[int]:
        """
        list of all indices in the pipeline parallel dimension

        Example:
            - we split a transformer (Embedding -> Layer 1 -> ... -> Layer n -> lm_head)
            vertically into pipe parallel ranks
            - A transformer with 2 layers on a pipe parallel of 4 may be layered out like:
                -> pipe_parallel_rank 0 gets Embedding
                -> pipe_parallel_rank 1 gets Layer 1
                -> pipe_parallel_rank 2 gets Layer 2
                -> pipe_parallel_rank 3 gets lm_head

                (this is simplified)

        """
        return list(range(self.config.pipe_parallel_size))

    @property
    def pipe_parallel_rank(self) -> int:
        return self._pipe_parallel_rank

    @property
    def previous_pipe_parallel_rank(self) -> Optional[int]:
        if self.is_first_pipe_parallel_rank:
            return None
        return self.pipe_parallel_rank - 1

    @property
    def next_pipe_parallel_rank(self) -> Optional[int]:
        if self.is_last_pipe_parallel_rank:
            return None
        return self.pipe_parallel_rank + 1

    @property
    def is_first_pipe_parallel_rank(self) -> bool:
        return self.pipe_parallel_rank == 0

    @property
    def is_last_pipe_parallel_rank(self) -> bool:
        return self.pipe_parallel_rank == (self.config.pipe_parallel_size - 1)

    @property
    def is_first_model_parallel_rank(self) -> bool:
        return self.model_parallel_rank == 0

    @property
    def is_io_rank(self) -> bool:
        """
        io ranks are only first and last pipe stages as well as only mp 0
        """

        return (
            self.is_first_pipe_parallel_rank or self.is_last_pipe_parallel_rank
        ) and self.is_first_model_parallel_rank

    @property
    def pipe_parallel_ranks(self) -> list[int]:
        return self._pipe_parallel_ranks  # type: ignore[return-value]

    @property
    def pipe_parallel_group(self) -> ProcessGroup:
        return self._pipe_parallel_group  # type: ignore[return-value]

    @property
    def data_parallel_indices(self) -> list[int]:
        """
        list of all indices in the data parallel dimension

        Example:
            - A transformer is split vertically and horizontally over
            pipe_parallel_ranks and model_parallel_intra_layer_ranks
            - For one instance of the transformer (pipe_parallel_size * model_parallel_size) gpus are used
            - Assume pipe_parallel_size is 4 and model_parallel_size is 2
            - This means one instance of the transformer uses 8 GPUs
            - If we have 16 GPUS, two copies (i.e. two data_parallel_ranks) can be loaded
        """
        return list(range(self.config.data_parallel_size))

    @property
    def data_parallel_rank(self) -> int:
        return self._data_parallel_rank

    @property
    def data_parallel_ranks(self) -> list[int]:
        return self._data_parallel_ranks  # type: ignore[return-value]

    @property
    def data_parallel_group(self) -> ProcessGroup:
        return self._data_parallel_group  # type: ignore[return-value]

    @property
    def model_parallel_indices(self) -> list[int]:
        """
        list of all indices in the model parallel dimension

        Example:
            - we split an attention heads over different model parallel ranks.
            - each model_parallel_rank will get a chunk of attention heads
            - in the case of 12 attention heads and 4 model_parallel_intra_layer_ranks
            the attention heads are layered out like this:
                -> model_parallel_rank 0 gets attention heads [0, 1, 2]
                -> model_parallel_rank 1 gets attention heads [3, 4, 5]
                -> model_parallel_rank 2 gets attention heads [6, 7, 8]
                -> model_parallel_rank 3 gets attention heads [9, 10, 11]
        """
        return list(range(self.config.model_parallel_size))

    @property
    def model_parallel_rank(self) -> int:
        return self._model_parallel_rank  # type: ignore[return-value]

    @property
    def model_parallel_ranks(self) -> list[int]:
        return self._model_parallel_ranks  # type: ignore[return-value]

    @property
    def model_parallel_group(self) -> ProcessGroup:
        return self._model_parallel_group  # type: ignore[return-value]

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Device not specified")
        return self._device

    def get_global_rank_group(
        self,
        pipe_parallel_rank: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        model_parallel_rank: Optional[int] = None,
        flatten: bool = True,
    ) -> list[int]:
        """
        Returns a list of all global ranks
        for any given combination of pipe_parallel_rank, data_parallel_rank, model_parallel_rank
        """
        ranks = self._layout_3D.clone()
        if pipe_parallel_rank is not None:
            ranks = ranks[pipe_parallel_rank : pipe_parallel_rank + 1, :, :]
        if data_parallel_rank is not None:
            ranks = ranks[:, data_parallel_rank : data_parallel_rank + 1, :]
        if model_parallel_rank is not None:
            ranks = ranks[:, :, model_parallel_rank : model_parallel_rank + 1]

        if flatten:
            return ranks.flatten().tolist()
        else:
            return ranks.tolist()

    @property
    def all_pipe_parallel_groups(self) -> list[list[int]]:
        """
        Returns a list of all global GPU rank groups
        for any combination of data_parallel_rank and model_parallel_rank

        One pipe_parallel_ranks combines all parameters of a transformer
        for a given model_parallel_rank and a data_parallel_rank.

        Example: (compare to docstring model_parallel_indices) attention heads [0, 1, 2]
        on all pipe_parallel_ranks of one instance (copy) of the transformer
        """

        pipe_parallel_ranks_all = list()
        for data_parallel_rank in self.data_parallel_indices:
            for model_parallel_rank in self.model_parallel_indices:
                pipe_parallel_ranks_all.append(
                    self.get_global_rank_group(
                        data_parallel_rank=data_parallel_rank,
                        model_parallel_rank=model_parallel_rank,
                    )
                )

        return pipe_parallel_ranks_all

    @property
    def all_data_parallel_groups(self) -> list[list[int]]:
        """
        Returns a list of all global GPU rank groups for any combination of model_parallel_rank and pipe_parallel_rank

        One data_parallel_ranks combines all parameters of a transformer
        for a given model_parallel_rank and a pipe_parallel_rank.
        Example: (compare to docstring model_parallel_intra_layer_ranks) attention heads [0, 1, 2]
        for pipe_parallel_rank 0 (i.e. first vertical split) for all data parallel copies
        """

        data_parallel_ranks_all = list()
        for pipe_parallel_rank in self.pipe_parallel_indices:
            for model_parallel_rank in self.model_parallel_indices:
                data_parallel_ranks_all.append(
                    self.get_global_rank_group(
                        model_parallel_rank=model_parallel_rank,
                        pipe_parallel_rank=pipe_parallel_rank,
                    )
                )

        return data_parallel_ranks_all

    @property
    def all_model_parallel_groups(self) -> list[list[int]]:
        """
        Returns a list of all global GPU rank groups for any combination of data_parallel_rank and pipe_parallel_rank

        One model_parallel_intra_layer_ranks combines all parameters of a transformer
        for a given pipeline rank and a data parallel rank.
        """

        model_parallel_intra_layer_ranks_all = []
        for pipe_parallel_rank in self.pipe_parallel_indices:
            for data_parallel_rank in self.data_parallel_indices:
                model_parallel_intra_layer_ranks_all.append(
                    self.get_global_rank_group(
                        data_parallel_rank=data_parallel_rank,
                        pipe_parallel_rank=pipe_parallel_rank,
                    )
                )

        return model_parallel_intra_layer_ranks_all

    def get_global_rank(
        self,
        pipe_parallel_rank: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        model_parallel_rank: Optional[int] = None,
    ) -> int:
        if pipe_parallel_rank is None:
            pipe_parallel_rank = self.pipe_parallel_rank
        if data_parallel_rank is None:
            data_parallel_rank = self.data_parallel_rank
        if model_parallel_rank is None:
            model_parallel_rank = self.model_parallel_rank

        return int(self._layout_3D[pipe_parallel_rank, data_parallel_rank, model_parallel_rank].item())
