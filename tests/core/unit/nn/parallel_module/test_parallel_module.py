import os
from typing import Callable

import pytest
import torch

from scaling.core.nn.parallel_module.parallel_module import _broadcast_parameters


def init_process(rank: int, size: int, return_dict: dict, expected: torch.nn.Parameter, fn: Callable) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_SLOT"] = str(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=size)

    fn(rank, size, return_dict, expected)


def run(rank: int, size: int, return_dict: dict, expected: torch.nn.Parameter) -> None:
    group = torch.distributed.new_group(ranks=list(range(size)))
    if rank == 0:
        param = torch.nn.Parameter(expected.detach().clone().to(device=f"cuda:{rank}"))
    else:
        param = torch.nn.Parameter(torch.zeros_like(expected, device=f"cuda:{rank}"))
    _broadcast_parameters(param, group)
    assert rank not in return_dict
    return_dict[rank] = torch.equal(param.to(device="cpu"), expected)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.unit
def test_broadcast_parameter(size: int) -> None:
    if size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs")
    expected = torch.nn.Parameter(torch.randn(40, 400, device="cpu"))

    processes = []
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    for rank in range(size):
        p = ctx.Process(target=init_process, args=(rank, size, return_dict, expected, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    casted_return_dict: dict = dict(return_dict)
    assert len(casted_return_dict) == size
    assert all(x for x in casted_return_dict.values())
