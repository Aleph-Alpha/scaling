# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Facebook, Deepmind Technologies,
# NYU, NEC Laboratories America and IDIAP Research Institute
#
# From PyTorch:
#
#     Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
#     Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
#     Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
#     Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
#     Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
#     Copyright (c) 2011-2013 NYU                      (Clement Farabet)
#     Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
#     Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
#     Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
#     From Caffe2:
#
#     Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
#     All contributions by Facebook:
#     Copyright (c) 2016 Facebook Inc.
#
#     All contributions by Google:
#     Copyright (c) 2015 Google Inc.
#     All rights reserved.
#
#     All contributions by Yangqing Jia:
#     Copyright (c) 2015 Yangqing Jia
#     All rights reserved.
#
#     All contributions by Kakao Brain:
#     Copyright 2019-2020 Kakao Brain
#
#     All contributions by Cruise LLC:
#     Copyright (c) 2022 Cruise LLC.
#     All rights reserved.
#
#     All contributions from Caffe:
#     Copyright(c) 2013, 2014, 2015, the respective contributors
#     All rights reserved.
#
#     All other contributions:
#     Copyright(c) 2015, 2016 the respective contributors
#     All rights reserved.
#
#     Caffe2 uses a copyright model similar to Caffe: each contributor holds
#     copyright over their contributions to Caffe2. The project versioning records
#     all such contribution and copyright details. If a contributor wants to further
#     mark their specific copyright on a particular contribution, they should
#     indicate their copyright solely in the commit message of the change when it is
#     committed.
#
#     All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable

import torch
from torch.utils.checkpoint import (
    _get_autocast_kwargs,
    get_device_states,
    set_device_states,
    weakref,
)

from scaling.core.topology import Topology


def _checkpoint_without_reentrant(
    function: Callable, topology: Topology, preserve_rng_state: bool = True, *args: tuple, **kwargs: dict
) -> Any:
    """Activation checkpointing, adapted from PyTorch (attribution see below)

    Activation checkpointing cannot ensure preservation of the rng state if the forward
    changes device placement. To fix that, we track rng states across devices in the
    `Topology`. This function extends PyTorch's checkpoint with the necessary bookkeeping.

    Checkpointing without re-entrant autograd

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)
            assert topology.has_model_parallel_constant_rng
            fwd_gpu_states_topology = topology.state_dict()["model_parallel_constant_rng"]  # type: ignore[index]

    # Custom class to be able to take weak references
    class Holder:
        pass

    # The Holder object for each of the saved object is saved directly on the
    # SavedVariable and is cleared when reset_data() is called on it. We MUST make
    # sure that this is the only object having an owning reference to ensure that
    # the Tensor stored in storage is deleted as soon as the corresponding SavedVariable
    # data is cleared.
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    def pack(x: Any) -> Holder:
        # TODO(varal7): Instead of returning abstract object, we can return things metadata (such as
        # size, device, ...) to catch certain cases of not deterministic behavior of the forward
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res

    def unpack(x: Any) -> torch.Tensor:
        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner: Any) -> None:
                nonlocal unpack_counter
                unpack_counter += 1
                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return
                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed: Any) -> torch.Tensor:
                raise RuntimeError("You are calling backwards on a tensor that is never exposed. Please open an issue.")

            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if preserve_rng_state and had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        set_device_states(fwd_gpu_devices, fwd_gpu_states)
                        # Next line was added.
                        assert topology._model_parallel_constant_rng is not None
                        topology._model_parallel_constant_rng.load_state_dict(fwd_gpu_states_topology)

                with torch.enable_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs), torch.cpu.amp.autocast(
                    **cpu_autocast_kwargs
                ), torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args, **kwargs)

        if x not in storage:
            raise RuntimeError(
                "Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                " re-computation being triggered in between, this is not currently supported. Please"
                " open an issue with details on your use case so that we can prioritize adding this."
            )

        return storage[x]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
        if torch.cuda._initialized and preserve_rng_state and not had_cuda_in_fwd:
            # Cuda was not initialized before running the forward, so we didn't
            # stash the CUDA state.
            raise RuntimeError(
                "PyTorch's CUDA state was initialized in the forward pass "
                "of a Checkpoint, which is not allowed. Please open an issue "
                "if you need this feature."
            )

    return output
