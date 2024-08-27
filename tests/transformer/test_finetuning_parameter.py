import os
from multiprocessing import Process, Queue, SimpleQueue
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from pydantic import ValidationError

from scaling.transformer.context.config import TrainingConfig
from scaling.transformer.model.model import _extract_parameters, _filter_by_param, _find_matching_param


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"finetune": True, "finetunable_parameters": ["a"], "parameters_exclude": ["b"]},
        {"finetune": True, "finetunable_parameters": ["a"]},
    ],
    ids=["Nothing is set", "Finetune is set and parameter", "Finetune is set and parameter 2"],
)
def test_config_raises_not(config):
    TrainingConfig(**config)
    TrainingConfig.from_dict(config)


def test_create_config_with_legacy_field_names():
    config = TrainingConfig(use_seperate_lr_on_embeddings=True)
    assert config.use_separate_lr_on_embeddings is True


@pytest.mark.parametrize(
    "config",
    [
        {"finetunable_parameters": ["a"]},
        {"finetune": True},
        {"finetunable_parameters": ["a"], "parameters_exclude": ["b"]},
    ],
    ids=[
        "Finetune is not set but parameter",
        "Finetune is set but parameter is not",
        "Finetune is not set but parameter 2",
    ],
)
def test_config_raises_error(config: dict):
    with pytest.raises(ValidationError):
        TrainingConfig(**config)
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict(config)


def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank)


def start_x_times(fn, size):
    processes = []

    for rank in range(size):
        p = Process(
            target=init_process,
            args=(
                rank,
                size,
                fn,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def test_not_matching():
    conf = TrainingConfig(
        finetune=True,
        finetunable_parameters=["summarization", "image_encoder"],
    )

    input_params = [
        [
            ("foo.not_relevant.bar", torch.tensor([1]), mock.MagicMock()),
            ("foo.summarization.bar", torch.tensor([1]), mock.MagicMock()),
        ],
        [
            ("foo.summarization.bar", torch.tensor([1]), mock.MagicMock()),
            ("foo.not_relevant_either.bar", torch.tensor([1]), mock.MagicMock()),
        ],
    ]

    exception_queue = Queue()

    def extract_parameters_of_rank(rank):
        try:
            _extract_parameters(conf, input_params[rank])
        except Exception as e:
            exception_queue.put((e, rank))

    start_x_times(extract_parameters_of_rank, 2)
    assert not exception_queue.empty()
    while not exception_queue.empty():
        e, _ = exception_queue.get()
        assert "Unmatched finetunable parameters: {'image_encoder'}" in str(e)


def test_matching():
    conf = TrainingConfig(
        finetune=True,
        finetunable_parameters=["summarization", "image_encoder"],
    )

    input_params = [
        [
            ("foo.not_relevant.bar", torch.tensor([1]), mock.MagicMock()),
            ("image_encoder.bar", torch.tensor([1]), mock.MagicMock()),
        ],
        [
            ("foo.summarization.bar", torch.tensor([1]), mock.MagicMock()),
            ("foo.not_relevant_either.bar", torch.tensor([1]), mock.MagicMock()),
        ],
    ]

    result_queue = SimpleQueue()

    def extract_parameters_of_rank(rank):
        e1, e2, e3 = _extract_parameters(conf, input_params[rank])
        e1 = [x[0] for x in e1]
        e2 = [x[0] for x in e2]
        e3 = [x[0] for x in e3]
        result_queue.put((rank, e1, e2, e3))

    start_x_times(extract_parameters_of_rank, 2)
    assert not result_queue.empty()
    while not result_queue.empty():
        rank, embedding_weight_decay, param_no_weight_decay, param_weight_decay = result_queue.get()
        if rank == 0:
            assert param_weight_decay == ["image_encoder.bar"]
        else:
            assert param_weight_decay == ["foo.summarization.bar"]
        assert embedding_weight_decay == param_no_weight_decay == []


def test_matching_with_exclusion():
    conf = TrainingConfig(
        finetune=True,
        finetunable_parameters=["summarization", "image_encoder"],
        parameters_exclude=["image_encoder.baz"],
    )

    input_params = [
        [
            ("foo.not_relevant.bar", torch.tensor([1]), mock.MagicMock()),
            ("image_encoder.bar", torch.tensor([1]), mock.MagicMock()),
            ("image_encoder.baz", torch.tensor([1]), mock.MagicMock()),
        ],
        [
            ("foo.summarization.bar", torch.tensor([1]), mock.MagicMock()),
            ("foo.not_relevant_either.bar", torch.tensor([1]), mock.MagicMock()),
        ],
    ]

    result_queue = SimpleQueue()

    def extract_parameters_of_rank(rank):
        e1, e2, e3 = _extract_parameters(conf, input_params[rank])
        e1 = [x[0] for x in e1]
        e2 = [x[0] for x in e2]
        e3 = [x[0] for x in e3]
        result_queue.put((rank, e1, e2, e3))

    start_x_times(extract_parameters_of_rank, 2)
    assert not result_queue.empty()
    while not result_queue.empty():
        rank, embedding_weight_decay, param_no_weight_decay, param_weight_decay = result_queue.get()
        if rank == 0:
            assert param_weight_decay == ["image_encoder.bar"]
        else:
            assert param_weight_decay == ["foo.summarization.bar"]
        assert embedding_weight_decay == param_no_weight_decay == []


@pytest.mark.parametrize(
    "data, result",
    [
        (["foo", "bar", "foo.bar"], "foo"),
        (["baz", "bar", "bay.bar"], "bar"),
        (["baz", "buz", "bay.bar"], None),
        (["baz", "buz", "foo.bar"], "foo.bar"),
    ],
    ids=["First match foo", "First match bar", "No match", "Last match"],
)
def test_find_matching_param(data: list[str], result: str):
    x = ("foo.bar", torch.tensor([1]), mock.MagicMock())
    assert result == _find_matching_param(x, data)


@pytest.fixture()
def fake_parameter_list():
    return [
        ("foo.bar", torch.tensor([1]), mock.MagicMock()),
        ("foo.baz", torch.tensor([1]), mock.MagicMock()),
        ("foo.buz", torch.tensor([1]), mock.MagicMock()),
    ]


def test_filter_by_param_everything(fake_parameter_list: list):
    assert _filter_by_param(["fuz", "foo"], fake_parameter_list) == []
    assert _filter_by_param(["foo"], fake_parameter_list) == []
    assert _filter_by_param(["fuz", "foo", "faz"], fake_parameter_list) == []


@pytest.mark.parametrize(
    "exclude,left_1, left_2",
    [("bar", 1, 2), ("baz", 0, 2), ("buz", 0, 1)],
    ids=["bar is removed", "baz is removed", "buz is removed"],
)
def test_filter_by_param(fake_parameter_list: list, exclude: str, left_1: int, left_2: int):
    expected = [fake_parameter_list[left_1], fake_parameter_list[left_2]]
    assert _filter_by_param([exclude], fake_parameter_list) == expected
