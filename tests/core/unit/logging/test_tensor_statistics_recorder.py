import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scaling.core import CoreParameterMeta, DataLoader, Topology
from scaling.core.logging import logger
from scaling.core.logging.tensor_statistics_recorder import (
    TENSOR_STATISTICS_FUNCTIONS,
    TensorStatistics,
    TensorStatisticsRecorder,
    statistics_fcts_for_name_pattern,
    tensor_tracker,
)
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.minimal import MinimalConfig, MinimalContext
from tests.core.minimal.data import MinimalDataset
from tests.core.minimal.model.model import (
    MinimalParallelModule,
    init_model,
    init_optimizer,
    loss_function,
    metrics_aggregation_fn,
)
from tests.core.utils import dist_launcher


@pytest.fixture
def sample_tensor():
    return torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.mark.cpu
def test_tensor_statistics_functions(sample_tensor):
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MEAN](sample_tensor) == pytest.approx(0.0)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.STD](sample_tensor) == pytest.approx(
        sample_tensor.numpy().std(ddof=1), rel=1e-2
    )
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.L1](sample_tensor) == pytest.approx(1.2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_STD](sample_tensor) == pytest.approx(
        np.abs(sample_tensor.numpy()).std(ddof=1), rel=1e-2
    )
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MAX](sample_tensor) == 2.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MIN](sample_tensor) == -2.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MEDIAN](sample_tensor) == 0.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_MAX](sample_tensor) == 2.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_MIN](sample_tensor) == 0.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_MEDIAN](sample_tensor) == 1.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.SKEW](sample_tensor) == pytest.approx(0.0, abs=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.KURTOSIS](sample_tensor) == pytest.approx(-1.3, abs=1e-1)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ENTROPY](torch.softmax(sample_tensor, dim=0)) == pytest.approx(
        1.0, rel=1e-2
    )
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_ZERO](sample_tensor) == pytest.approx(0.2, abs=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_UNDERFLOW_E4M3](sample_tensor) == pytest.approx(
        0.2, abs=1e-2
    )
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_UNDERFLOW_E5M2](sample_tensor) == pytest.approx(
        0.2, abs=1e-2
    )
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_POSINF](sample_tensor) == 0.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_NEGINF](sample_tensor) == 0.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.PCT_NAN](sample_tensor) == 0.0
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MAX_SPARSITY](sample_tensor) == pytest.approx(1.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.NORM_SPARSITY](sample_tensor) == pytest.approx(1.1785, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_1](sample_tensor) == pytest.approx(-1.96, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_10](sample_tensor) == pytest.approx(-1.6, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_25](sample_tensor) == pytest.approx(-1.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_75](sample_tensor) == pytest.approx(1.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_90](sample_tensor) == pytest.approx(1.6, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.QUANTILE_99](sample_tensor) == pytest.approx(1.96, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_1](sample_tensor) == pytest.approx(0.04, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_10](sample_tensor) == pytest.approx(0.4, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_25](sample_tensor) == pytest.approx(1.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_75](sample_tensor) == pytest.approx(2.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_90](sample_tensor) == pytest.approx(2.0, rel=1e-2)
    assert TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_QUANTILE_99](sample_tensor) == pytest.approx(2.0, rel=1e-2)


@pytest.mark.cpu
def test_statistics_fcts_for_name_pattern():
    name_pattern = "the_answer_is_42"

    statistics_config = [("*", [TensorStatistics.MEAN, TensorStatistics.STD])]
    statistics_ftcs = statistics_fcts_for_name_pattern(statistics_config, name_pattern)
    assert statistics_ftcs == {
        "mean": TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MEAN],
        "std": TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.STD],
    }

    statistics_config = [
        ("the_*", [TensorStatistics.ABS_MAX]),
        ("*", [TensorStatistics.ABS_MIN]),
    ]
    statistics_ftcs = statistics_fcts_for_name_pattern(statistics_config, name_pattern)
    assert statistics_ftcs == {
        "abs_max": TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.ABS_MAX],
    }

    statistics_config = [
        ("the_answer_is_41", [TensorStatistics.ABS_MEDIAN]),
        ("*", [TensorStatistics.MEDIAN]),
    ]
    statistics_ftcs = statistics_fcts_for_name_pattern(statistics_config, name_pattern)
    assert statistics_ftcs == {
        "median": TENSOR_STATISTICS_FUNCTIONS[TensorStatistics.MEDIAN],
    }


@pytest.mark.cpu
def test_recording_pytorch_model():
    torch.manual_seed(42)

    # instantiate small model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear_1.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear_1.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear_1.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear_1.bias",
            )
            self.linear_2 = torch.nn.Linear(42, 42, bias=True)
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear_2.weight,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear2.weight",
            )
            CoreParameterMeta.register_on_parameter(
                parameter=self.linear_2.bias,
                is_model_parallel=False,
                layer_index=0,
                parameter_name="linear_2.bias",
            )

        def forward(self, x):
            x = self.linear_1(x)
            tensor_tracker.trace_tensor(x, "hidden_1_tensor")
            x = self.linear_2(x)
            # note this will not be logged since config has not matching pattern
            tensor_tracker.trace_tensor(x, "hidden_2_tensor")

            return x.sum()

    model = Model()

    config_dict: Dict[str, Any] = dict()
    config_dict["topology"] = dict()
    config_dict["topology"]["model_parallel_size"] = 1
    config_dict["topology"]["pipe_parallel_size"] = 1
    config_dict["topology"]["world_size"] = 1
    config_dict["topology"]["global_rank"] = 0
    config_dict["topology"]["local_slot"] = 0
    config_dict["topology"]["micro_batch_size"] = 2
    config_dict["topology"]["gradient_accumulation_steps"] = 1
    config_dict["trainer"] = dict()
    config_dict["trainer"]["tensor_statistics_recorder_config"] = {
        "interval": 5,
        "statistics": [
            ["activation_statistics/*linear_1*", ["max_sparsity"]],
            ["activation_statistics/*linear_2*", ["norm_sparsity"]],
            ["tensor_statistics/*hidden_1*", ["mean"]],
        ],
        "include_module_type": True,
    }

    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = MinimalContext(config=config, topology=topology)

    mock_logger = MagicMock()

    tensor_statistics_recorder = TensorStatisticsRecorder(
        config=config.trainer.tensor_statistics_recorder_config,
        context=context,
        model=model,
        logger=mock_logger,
    )

    # Simulate fwd/bwd pass
    x = torch.randn((2, 42))
    x.requires_grad = True

    with tensor_statistics_recorder.trace():
        y = model.forward(x)
        y.backward()

    assert mock_logger.log_metrics.called
    assert mock_logger.log_metrics.call_count == 6
    assert mock_logger.log_metrics.call_args_list[0].kwargs["metrics"] == {
        "activation_statistics/layers.1.Linear.linear_1_forward_max_sparsity": pytest.approx(
            3.224484443664551, rel=1e-2
        )
    }
    assert mock_logger.log_metrics.call_args_list[1].kwargs["metrics"] == {
        "tensor_statistics/counter.0.hidden_1_tensor_forward_mean": pytest.approx(-0.056191034615039825, rel=1e-2)
    }
    assert mock_logger.log_metrics.call_args_list[2].kwargs["metrics"] == {
        "activation_statistics/layers.2.Linear.linear_2_forward_norm_sparsity": pytest.approx(
            1.2611907720565796, rel=1e-2
        )
    }
    assert mock_logger.log_metrics.call_args_list[3].kwargs["metrics"] == {
        "activation_statistics/layers.2.Linear.linear_2_backward_norm_sparsity": pytest.approx(1.0, rel=1e-2)
    }
    assert mock_logger.log_metrics.call_args_list[4].kwargs["metrics"] == {
        "tensor_statistics/counter.0.hidden_1_tensor_backward_mean": pytest.approx(-0.05661824345588684, rel=1e-2)
    }
    assert mock_logger.log_metrics.call_args_list[5].kwargs["metrics"] == {
        "activation_statistics/layers.1.Linear.linear_1_backward_max_sparsity": pytest.approx(
            2.503053903579712, rel=1e-2
        )
    }
    assert mock_logger.log_metrics.call_args_list[5].kwargs["step"] == 0


def run_distributed(save_dir_path: Path, model_parallel_size: int, pipe_parallel_size: int, **kwargs):
    torch.manual_seed(42)

    launch_config = LaunchConfig.from_launcher_args()
    config_dict: Dict[str, Any] = dict()
    config_dict["topology"] = {
        "model_parallel_size": model_parallel_size,
        "pipe_parallel_size": pipe_parallel_size,
        "world_size": launch_config.world_size,
        "global_rank": launch_config.global_rank,
        "local_slot": launch_config.local_slot,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
    }
    config_dict["trainer"] = dict()
    config_dict["trainer"]["tensor_statistics_recorder_config"] = {
        "interval": 5,
        "statistics": [
            ["activation_statistics/layers.1*", ["max_sparsity"]],
            ["parameter_statistics/layers.1*", ["abs_max"]],
            ["activation_statistics/layers.2*", ["norm_sparsity"]],
            ["parameter_statistics/layers.2*", ["abs_min"]],
        ],
    }
    config_dict["logger"] = {
        "log_dir": str(save_dir_path),
        "use_tensorboard": True,
        "tensorboard_ranks": {
            "data_parallel_rank": 0,
            "model_parallel_rank": 0,
        },
    }

    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = MinimalContext(config=config, topology=topology)
    context.initialize(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        seed=42,
    )

    model: MinimalParallelModule = init_model(context)
    optimizer = init_optimizer(context, model)
    dataset = MinimalDataset(seed=42)
    dataloader = DataLoader(seed=42, consumed_samples=0, dataset=dataset, topology=topology)

    tensor_statistics_recorder = TensorStatisticsRecorder(
        config=config.trainer.tensor_statistics_recorder_config,  # type: ignore
        context=context,
        model=model,
    )

    # Monkey-patch logger to record call arguments
    call_args_list = []

    def mock_log_metrics(*args, **kwargs):
        call_args_list.append({"args": args, "kwargs": kwargs})

    with patch.object(logger, "log_metrics", side_effect=mock_log_metrics):
        # Simulate train step
        with tensor_statistics_recorder.trace():
            _ = model.train_step(
                dataloader=dataloader,
                optimizer=optimizer,
                sync_batch_to_model_parallel=MinimalDataset.sync_batch_to_model_parallel,
                loss_function=loss_function,  # type: ignore
                metrics_aggregation_fn=metrics_aggregation_fn,
            )

    # Simulate train step
    with tensor_statistics_recorder.trace():
        _ = model.train_step(
            dataloader=dataloader,
            optimizer=optimizer,
            sync_batch_to_model_parallel=MinimalDataset.sync_batch_to_model_parallel,
            loss_function=loss_function,  # type: ignore
            metrics_aggregation_fn=metrics_aggregation_fn,
        )

    # Store all metrics that were logged into per-process files (will be loaded by master process)
    metrics_file_path = save_dir_path / f"metrics_{os.getpid()}.log"
    with open(metrics_file_path, "w") as f:
        for i in range(len(call_args_list)):
            f.write(list(call_args_list[i]["kwargs"]["metrics"].keys())[0] + "\n")


@pytest.mark.parallel_module
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size",
    [(1, 1), (1, 2)],
)
def test_recording_distributed(tmp_path: Path, model_parallel_size: int, pipe_parallel_size: int):
    world_size = model_parallel_size * pipe_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    _ = dist_launcher(
        run_func=run_distributed,
        world_size=world_size,
        master_port=find_free_port(),
        save_dir_path=tmp_path,
        model_parallel_size=model_parallel_size,
        pipe_parallel_size=pipe_parallel_size,
    )

    expected_metric_names = [
        "parameter_statistics/layers.1.linear.weight_forward_abs_max",
        "parameter_statistics/layers.1.linear.bias_forward_abs_max",
        "parameter_statistics/layers.2.linear.weight_forward_abs_min",
        "parameter_statistics/layers.2.linear.bias_forward_abs_min",
        "parameter_statistics/layers.1.linear.weight_backward_abs_max",
        "parameter_statistics/layers.1.linear.bias_backward_abs_max",
        "parameter_statistics/layers.2.linear.bias_backward_abs_min",
        "parameter_statistics/layers.2.linear.weight_backward_abs_min",
        "activation_statistics/layers.1.linear_forward_max_sparsity",
        "activation_statistics/layers.2.linear_forward_norm_sparsity",
        "activation_statistics/layers.1.linear_backward_max_sparsity",
        "activation_statistics/layers.2.linear_backward_norm_sparsity",
    ]

    # Load all metrics that were stored in subprocesses
    metrics_logged = []
    for metrics_file in tmp_path.glob("metrics_*.log"):
        with open(metrics_file, "r") as f:
            metrics_logged.extend(f.read().splitlines())

    for metric_name in expected_metric_names:
        assert metric_name in metrics_logged
