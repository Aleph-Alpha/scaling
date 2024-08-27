import random
from pathlib import Path
from typing import Any, Optional, TypedDict, TypeVar

import numpy as np
import torch

from scaling.core.config import BaseConfig
from scaling.core.topology import Topology, TopologyState

try:
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = None  # type: ignore
    DeterminedProfilerAgent = None  # type: ignore


class ContextState(TypedDict):
    iterations: int
    consumed_samples: int
    consumed_samples_evaluation: int
    random_rng_state: tuple
    np_rng_state: dict[str, Any]
    torch_rng_state: torch.Tensor
    torch_cuda_rng_state: torch.Tensor
    topology: Optional[TopologyState]


class BaseContext:
    """
    Context containing information for a train or inference process
    The config is regarded to be immutable.
    """

    def __init__(
        self,
        config: BaseConfig,
        topology: Topology,
    ) -> None:
        self.config = config
        self.topology = topology

        self.iterations = 0
        self.consumed_samples = 0
        self.consumed_samples_evaluation = 0

    def initialize(
        self,
        master_addr: str,
        master_port: str,
        torch_distributed_timeout_minutes: int = 20,
        seed: int = 42,
        distributed: bool = True,
    ) -> None:
        """
        Initialization of context state.
        This includes:
            - topology torch distributed
            - seeds
        """

        # initialize distributed
        if self.topology is not None:
            if distributed:
                self.topology.initialize_distributed(
                    master_addr=master_addr,
                    master_port=master_port,
                    torch_distributed_timeout_minutes=torch_distributed_timeout_minutes,
                    seed=seed,
                )
            else:
                self.topology.initialize_device()

        # set seed
        if self.topology is not None:
            assert self.topology.config.global_rank is not None
            seed = seed + self.topology.config.global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed(seed)

    def step(self) -> None:
        self.iterations += 1
        assert self.topology.config.global_batch_size is not None
        self.consumed_samples += self.topology.config.global_batch_size

    def state_dict(self) -> ContextState:
        return {
            "iterations": self.iterations,
            "consumed_samples": self.consumed_samples,
            "consumed_samples_evaluation": self.consumed_samples_evaluation,
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(),
            "topology": self.topology.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.iterations = state_dict["iterations"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.consumed_samples_evaluation = state_dict.get("consumed_samples_evaluation", 0)
        random.setstate(state_dict["random_rng_state"])
        np.random.set_state(state_dict["np_rng_state"])
        torch.set_rng_state(state_dict["torch_rng_state"])
        torch.cuda.set_rng_state(state_dict["torch_cuda_rng_state"])
        self.topology.load_state_dict(state_dict.get("topology"))

    def save_checkpoint(self, dir: Path | str) -> None:
        """
        Save the context state to a directory.
        Assumption is that there are no name collisions of files.
        """

        dir = Path(dir)
        if self.topology.config.global_rank == 0:
            self.config.save(dir / "config.yml")
        torch.save(
            self.state_dict(),
            str(dir / f"context_global_rank_{self.topology.config.global_rank}.pt"),
        )

    def load_checkpoint(self, dir: Path | str) -> None:
        """
        Load the state into an already initialized context
        """
        dir = Path(dir)

        # load checkpoint file if exists
        checkpoint_file = dir / f"context_global_rank_{self.topology.config.global_rank}.pt"
        if checkpoint_file.is_file():
            state_dict = torch.load(str(checkpoint_file))
            self.load_state_dict(state_dict)

        # if the context checkpoint does not exist, new global ranks are in play
        # in this case iterations and consumed samples need to be synced
        if self.topology.is_distributed_initialized:
            if self.topology.config.global_rank == 0:
                t = torch.tensor(
                    [
                        self.iterations,
                        self.consumed_samples,
                        self.consumed_samples_evaluation,
                    ]
                ).cuda()
            else:
                t = torch.tensor([0, 0, 0]).cuda()

            torch.distributed.all_reduce(
                t,
                op=torch.distributed.ReduceOp.MAX,
                group=torch.distributed.group.WORLD,
            )

            self.iterations = int(t[0].item())
            self.consumed_samples = int(t[1].item())
            self.consumed_samples_evaluation = int(t[2].item())


BaseContextGeneric = TypeVar("BaseContextGeneric", bound=BaseContext)


class DeterminedBaseContext(BaseContext):
    def __init__(
        self,
        config: BaseConfig,
        topology: Topology,
    ) -> None:
        super().__init__(config=config, topology=topology)
        self.determined_context: Optional[DeterminedContext] = None  # type: ignore
        self.determined_profiler: Optional[DeterminedProfilerAgent] = None  # type: ignore
        self._use_determined = False

    def initialize_with_determined(
        self,
        master_addr: str,
        master_port: str,
        determined_context: DeterminedContext,  # type: ignore
        determined_profiler: Optional[DeterminedProfilerAgent],  # type: ignore
        torch_distributed_timeout_minutes: int = 20,
        seed: int = 42,
        distributed: bool = True,
    ) -> None:
        super().initialize(
            master_addr,
            master_port,
            torch_distributed_timeout_minutes,
            seed,
            distributed,
        )
        self.determined_context = determined_context
        self.determined_profiler = determined_profiler
        self._use_determined = True
