import os
import random
import string
from datetime import timedelta
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_map

from scaling.core.nn.parallel_module import PipeCommunicator
from scaling.core.nn.parallel_module.communicator import (
    CommunicationMetaBase,
    dump_to_pickle_tensor,
    map_loads_from_pickle_tensor_fn,
)
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.utils import dist_launcher


class DummySettings:
    def __init__(self, name: str, some_int_list: list):
        self.name = name
        self.some_int_list = some_int_list


DUMMY_DATA_1 = torch.tensor([[1.0]])

DUMMY_DATA_2 = ((torch.tensor([[1.0]]),),)

DUMMY_DATA_3 = {"data": torch.tensor([[1.0]])}

DUMMY_DATA_4 = {
    "data": torch.tensor([[1.0]]),
    "more_data": torch.tensor([[1.0]]),
}

DUMMY_DATA_5 = (torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[42]]))

DUMMY_DATA_6 = {
    "data": torch.tensor([[1.0]]),
    "more_data_non_fp": torch.tensor([[42]]),
}

DUMMY_DATA_7 = {
    "data": torch.tensor([[1.0]]),
    "more_data": torch.tensor([[1.0]]),
    "more_data_non_fp": torch.tensor([[42]]),
}

DUMMY_DATA_8 = {
    "a": [torch.zeros(size=(5, 33, 7)), torch.zeros(size=(55, 55, 55, 1))],
    "b": torch.zeros(size=(1,)),
    "c": {"d": torch.zeros(size=(5, 5)), "e": torch.zeros(size=(115,))},
    "d": [True, "THIS IS A TEXT"],
    "settings": DummySettings(name="THIS IS A AMAZING NAME", some_int_list=[0, -55, 5, 5, 66]),
}

DUMMY_DATA_9 = [
    False,
    "TRUE",
    {"a": torch.zeros(size=(1,)), "b": torch.zeros(size=(1,))},
    np.zeros((2, 2, 5)),
]


def run_any_communication(return_dict: dict, dummy_data: Any, use_continuous_recommunication: bool):
    device, rank = _init_distributed()

    if rank == 0:

        def map_to_fn(x: Any):
            if torch.is_tensor(x):
                return x.clone().detach().to(device)
            else:
                return x

        com = _create_pipe_communicator(device, use_continuous_recommunication, False)
        dummy_data = tree_map(map_to_fn, dummy_data)

        com.send_data(data=dummy_data, target_global_rank=1)
        com.send_data(data=dummy_data, target_global_rank=1)
        com.send_data(data=dummy_data, target_global_rank=1)

    elif rank == 1:
        com = _create_pipe_communicator(device, use_continuous_recommunication, True)

        com.recv_data(origin_global_rank=0)
        com.recv_data(origin_global_rank=0)
        com.recv_data(origin_global_rank=0)

    else:
        raise RuntimeError


def _init_distributed():
    launch_config = LaunchConfig.from_launcher_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(launch_config.global_rank)
    rank = launch_config.global_rank
    device = torch.device("cuda")
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=2,
        rank=rank,
        init_method=f"tcp://{launch_config.master_addr}:{launch_config.master_port}",
        timeout=timedelta(minutes=1),
    )
    return device, rank


def create_random_dummy_data():
    a = torch.rand(size=(5, 3, 3), dtype=torch.float)
    b = torch.rand(size=(1,), dtype=torch.bfloat16)
    c = torch.rand(size=(7, 7), dtype=torch.bfloat16)

    d = "".join((random.choice(string.ascii_letters) for i in range(random.randint(12, 16))))
    _e = [random.choice(list(range(-50, 50))) for i in range(random.randint(12, 16))]
    e = DummySettings(name=d, some_int_list=_e)

    return [a, (b, c), d, e]


def run_const_communication(return_dict: dict):
    device, rank = _init_distributed()
    if rank == 0:
        dummy_data = create_random_dummy_data()

        com = _create_pipe_communicator(device, True, False)

        def map_to_fn(x: Any):
            if torch.is_tensor(x):
                return x.clone().detach().to(device)
            else:
                return x

        dummy_data = tree_map(map_to_fn, dummy_data)

        _, did_reset = com._send_meta_data(data=dummy_data, target_global_rank=1)
        assert did_reset, "needs to reset on the first pass"

        for _ in range(5):
            dummy_data = create_random_dummy_data()
            dummy_data = tree_map(map_to_fn, dummy_data)

            _, did_reset = com._send_meta_data(data=dummy_data, target_global_rank=1)
            assert not did_reset, "should not reset, on the second pass"

        dummy_data = create_random_dummy_data()
        dummy_data.append("FFFGGG")
        dummy_data = tree_map(map_to_fn, dummy_data)
        _, did_reset = com._send_meta_data(data=dummy_data, target_global_rank=1)
        assert did_reset, "needs to reset on the first pass"

        for _ in range(5):
            dummy_data = create_random_dummy_data()
            dummy_data.append("FFFGGG")
            dummy_data = tree_map(map_to_fn, dummy_data)

            _, did_reset = com._send_meta_data(data=dummy_data, target_global_rank=1)
            assert not did_reset, "should not reset, on the second pass"

    elif rank == 1:
        com = _create_pipe_communicator(device, True, True)

        did_reset = com._recv_meta_data(origin_global_rank=0)
        assert did_reset, "needs to reset on the first pass"

        for _ in range(5):
            did_reset = com._recv_meta_data(origin_global_rank=0)
            assert not did_reset, "should not reset, on the second pass"

        did_reset = com._recv_meta_data(origin_global_rank=0)
        assert did_reset, "needs to reset on the first pass"

        for _ in range(5):
            did_reset = com._recv_meta_data(origin_global_rank=0)
            assert not did_reset, "should not reset, on the second pass"

    else:
        raise RuntimeError


def _create_pipe_communicator(device, use_continuous_recommunication, recv_data):
    com = PipeCommunicator(
        local_device=device,
        recv_grads=False,
        recv_data=recv_data,
        use_continuous_recommunication=use_continuous_recommunication,
    )
    com.reset_communication_meta()
    return com


@pytest.mark.parallel_module
@pytest.mark.parametrize(
    "dummy_input",
    [
        DUMMY_DATA_1,
        DUMMY_DATA_2,
        DUMMY_DATA_3,
        DUMMY_DATA_4,
        DUMMY_DATA_5,
        DUMMY_DATA_6,
        DUMMY_DATA_7,
        DUMMY_DATA_8,
        DUMMY_DATA_9,
    ],
)
@pytest.mark.parametrize("use_continuous_recommunication", [True, False])
def test_any_communication(dummy_input, use_continuous_recommunication):
    _ = dist_launcher(
        run_func=run_any_communication,
        dummy_data=dummy_input,
        use_continuous_recommunication=use_continuous_recommunication,
        world_size=2,
        master_port=find_free_port(),
    )


@pytest.mark.parallel_module
def test_const_communication():
    _ = dist_launcher(
        run_func=run_const_communication,
        world_size=2,
        master_port=find_free_port(),
    )


@pytest.mark.parallel_module
@pytest.mark.cpu
def test_pickle():
    dummy_meta = CommunicationMetaBase(is_tensor=False, tensor_dtype=None, tensor_shape=None, requires_grad=None)

    a = "HHISDFUGS"
    t_a = dump_to_pickle_tensor(x=a, max_buffer_size=None)

    t_a_len = t_a.size(0)

    b = "HHISDFUGS_x"
    t_b = dump_to_pickle_tensor(x=b, max_buffer_size=t_a_len)

    t_b_len = t_b.size(0)

    assert t_a_len == t_b_len, "len should not change"

    c = "HHISDFUGS_HHISDFUGS_HHISDFUGS_HHISDFUGS_HHISDFUGS_HHISDFUGS_x"
    t_c = dump_to_pickle_tensor(x=c, max_buffer_size=t_a_len)

    t_c_len = t_c.size(0)

    assert t_a_len != t_c_len, "len should change"

    _a = map_loads_from_pickle_tensor_fn(t=t_a, meta=dummy_meta)
    _b = map_loads_from_pickle_tensor_fn(t=t_b, meta=dummy_meta)
    _c = map_loads_from_pickle_tensor_fn(t=t_c, meta=dummy_meta)

    assert a == _a, "load needs to be the same as the origen."
    assert b == _b, "load needs to be the same as the origen."
    assert c == _c, "load needs to be the same as the origen."
