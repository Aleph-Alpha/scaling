from typing import Any, Dict

import pytest
import torch

from scaling.core.utils.port import find_free_port

from ..minimal import MinimalConfig, main
from ..utils import dist_launcher


def run_test_training(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    metrics_list = main(overwrite_config=config_dict, return_metrics=True)

    if config_dict["topology"]["global_rank"] == 0:
        assert metrics_list is not None
        return_dict["losses"] = [metric["training/loss"] for metric in metrics_list]


@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (1, 2, 4),
        (2, 1, 2),
    ],
)
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16"), (False, "float32")])
@pytest.mark.parametrize("activation_checkpointing_type", ["every_pipe_stage", "every_layer"])
def test_activation_checkpointing(
    tmp_path_factory,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    enable_loss_scaling: bool,
    precision: str,
    activation_checkpointing_type: str,
):
    """
    Tests if different activation checkpointing types produce the same loss during training
    as training with disabled activation checkpointing
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    # create two tmp cache dir folders for activation checkpointing enabled and disabled
    cache_dir_activation_checkpointing = tmp_path_factory.mktemp(
        f"test_activation_checkpointing_{model_parallel_size}_{pipe_parallel_size}_{world_size}_{enable_loss_scaling}_{precision}_{activation_checkpointing_type}"
    )
    cache_dir_disabled_activation_checkpointing = tmp_path_factory.mktemp(
        f"test_activation_checkpointing_{model_parallel_size}_{pipe_parallel_size}_{world_size}_{enable_loss_scaling}_{precision}_disabled"
    )

    # set up config with activation checkpointing
    config_dict_activation_checkpointing: Dict[str, Any] = {
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing_type": activation_checkpointing_type,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {"enable": enable_loss_scaling},
        },
        "learning_rate_scheduler": {
            "learning_rate": 0.1,
            "learning_rate_minimum": 0.0,
            "learning_rate_decay_style": "cosine",
            "learning_rate_warmup_steps": 2,
            "learning_rate_decay_iters": 10,
        },
        "trainer": {
            "save_dir": str(cache_dir_activation_checkpointing),
            "save_interval": 10,
            "load_dir": str(cache_dir_activation_checkpointing),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {
            "log_level": "debug",
            "log_dir": str(cache_dir_activation_checkpointing / "logs"),
        },
        "training": {
            "precision": precision,
        },
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
    }
    config_activation_checkpointing: MinimalConfig = MinimalConfig.from_dict(config_dict_activation_checkpointing)
    # set up config without activation checkpointing
    config_dict_disabled_activation_checkpointing = config_dict_activation_checkpointing
    config_dict_disabled_activation_checkpointing["topology"]["activation_checkpointing_type"] = "disabled"
    config_dict_disabled_activation_checkpointing["trainer"]["save_dir"] = str(
        cache_dir_disabled_activation_checkpointing
    )
    config_dict_disabled_activation_checkpointing["trainer"]["load_dir"] = str(
        cache_dir_disabled_activation_checkpointing
    )
    config_dict_disabled_activation_checkpointing["logger"]["log_dir"] = str(
        cache_dir_disabled_activation_checkpointing / "logs"
    )
    config_disabled_activation_checkpointing: MinimalConfig = MinimalConfig.from_dict(
        config_dict_disabled_activation_checkpointing
    )

    # Train with activation checkpointing for up to 10 steps
    return_dict_with_activation_checkpointing_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_activation_checkpointing.as_dict(),
    )

    # Train without activation checkpointing for up to 10 steps
    config_dict_activation_checkpointing["trainer"]["assert_checkpoint_loaded"] = True
    config_activation_checkpointing = MinimalConfig.from_dict(config_dict_activation_checkpointing)
    return_dict_without_activation_checkpointing_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_disabled_activation_checkpointing.as_dict(),
    )

    assert (
        return_dict_with_activation_checkpointing_trained_model["losses"]
        == return_dict_without_activation_checkpointing_trained_model["losses"]
    ), (
        f"loss changed without activation checkpointing; "
        f"expected {return_dict_without_activation_checkpointing_trained_model['losses']}, "
        f"got {return_dict_with_activation_checkpointing_trained_model['losses']}"
    )
