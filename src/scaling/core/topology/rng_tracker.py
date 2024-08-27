# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Microsoft Corporation
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Iterator, TypedDict

import torch
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager


class RngTrackerState(TypedDict):
    seed: int
    state: torch.Tensor


def _set_cuda_rng_state(new_state: torch.Tensor, device: int | str | torch.device = -1) -> None:
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb() -> None:
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)  # type: ignore

    else:
        # newer PyTorch
        if device == -1:
            t_device = torch.device("cuda")
        elif isinstance(device, str):
            t_device = torch.device(device)
        elif isinstance(device, int):
            t_device = torch.device("cuda", device)
        else:
            assert isinstance(device, torch.device)
            t_device = device

        def cb() -> None:
            idx = t_device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStateTracker:
    def __init__(self, seed: int):
        self.seed = seed

        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.state = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    def state_dict(self) -> RngTrackerState:
        return {"seed": self.seed, "state": self.state.clone().detach()}

    def load_state_dict(self, state_dict: RngTrackerState) -> None:
        self.seed = state_dict["seed"]
        self.state = state_dict["state"]

    @contextlib.contextmanager
    def fork(self) -> Iterator[None]:
        """
        Fork the cuda rng state, perform operations, and exit with
        the original state.
        """
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.state)
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.state = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)
