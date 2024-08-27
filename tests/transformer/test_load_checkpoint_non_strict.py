from pathlib import Path
from typing import Dict

import pytest
import torch

from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer import TransformerConfig
from scaling.transformer.train import main

from .utils import dist_launcher

EPS = 1e-5


def run_test_training(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    losses = main(launch_config=launch_config, overwrite_config=config_dict, return_metrics=True)
    return_dict["losses"] = losses


@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 2, 4),
        (2, 2, 4),
    ],
)
def test_load_checkpoint_non_strict(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    micro_batch_size: int = 2
    gradient_accumulation_steps = 1
    enable_loss_scaling: bool = False
    precision: str = "float32"
    legacy_dataset: bool = False

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {
                "enable": enable_loss_scaling,
                "initial_scale": 16,  # set low initial loss scale to actually perform a train step in this short test
            },
        },
        "learning_rate_scheduler": {
            "learning_rate": 0.1,
            "learning_rate_minimum": 0.0,
            "learning_rate_decay_style": "cosine",
            "learning_rate_warmup_steps": 2,
            "learning_rate_decay_iters": 10,
        },
        "trainer": {
            "save_dir": str(tmp_path),
            "save_interval": 6,
            "load_dir": str(tmp_path),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
        "training": {},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 10},
        "data": {
            "data_prefixes": (
                [Path(__file__).parents[0] / "files" / "dataset" / "legacy" / "enron_text_document_100"] * 2
                if legacy_dataset
                else [Path(__file__).parents[0] / "files" / "dataset" / "data"]
            ),
            "legacy_dataset": legacy_dataset,
            "blended_dataset": {"cache_directory": tmp_path},
        },
        "transformer_architecture": {
            "vocab_size": 128000,
            "sequence_length": 8,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_layers": 2,
            "precision": precision,
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
        },
    }
    # config_dict["optimizer"]["zero"] = True
    config = TransformerConfig.from_dict(config_dict)

    # Train up to 3 steps
    # This function will checkpoint after 2 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume and one that was
    # trained continuously
    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    # Resume model training from the previous checkpoint at 2 steps.
    # Train up to 3 steps after loading from checkpoint
    # Step 2 to 3 should have the same losses for both trainings
    # config_dict["optimizer"]["zero"] = True
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["transformer_architecture"]["softprompt_config"] = {"name": "summarization", "n_tokens": 2}
    config_dict["transformer_architecture"]["adapter_config"] = {
        "name": "image_encoder",
        "attention_downsampling_factor": 1.0,
        "mlp_downsampling_factor": 1.0,
    }

    config_dict["trainer"]["load_optimizer_states"] = False
    added_params = [
        "softprompt_summarization",
        "attn_adapter_image_encoder.dense_in.weight",
        "attn_adapter_image_encoder.dense_out.weight",
        "mlp_adapter_image_encoder.dense_in.weight",
        "mlp_adapter_image_encoder.dense_out.weight",
    ]
    config_dict["trainer"]["allowed_missing_keys_in_checkpoint"] = added_params
    config_dict["training"]["finetune"] = True
    config_dict["training"]["finetunable_parameters"] = [
        "summarization",
        "image_encoder",
    ]

    config_loaded = TransformerConfig.from_dict(config_dict)
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )
    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["losses"][-4:],
        return_dict_resumed_trained_model["losses"],
    ):
        # the train loss has to be different because we added adapters and softprompt
        # also note that the sequence length is shortened
        # to make room for softprompt tokens, so the data is also different
        if loss_original["training/loss"] is not None and loss_resumed["training/loss"] is not None:
            different_losses = abs(loss_original["training/loss"] - loss_resumed["training/loss"]) > EPS
            assert (
                different_losses
            ), "loss did not change after continuing training from a checkpoint and adding adapters and softprompt"
        else:
            assert model_parallel_size > 1, "loss_original and loss_resumed can only be None if model_parallel_size > 1"
