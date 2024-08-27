import os
from pathlib import Path
from typing import Dict

import pytest
import torch

from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from tests.transformer.test_training import run_test_training

from .utils import dist_launcher


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
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("num_local_attention_heads", [16, 4])
@pytest.mark.parametrize("local_attention_window_size", [64, 256])
def test_transformer_local_attention_training(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_local_attention_heads: int,
    local_attention_window_size: int,
):
    """
    End-to-end test for training with and without flash attention and expecting the same loss

    Includes:
        - Setup of model in a distributed environment
        - Training without Flash Attention
        - Training with Flash Attention
        - Compare losses
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    print("cache_dir", tmp_path)

    config_dict: Dict = {
        "runner": {"use_determined": False},
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "activation_checkpointing_type": "disabled",
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {
                "enable": False,
                "initial_scale": 16,  # set low initial loss scale to actually perform a train step in this short test
            },
            "zero": True,
        },
        "learning_rate_scheduler": {
            "learning_rate": 0.1,
            "learning_rate_minimum": 0.0,
            "learning_rate_decay_style": "cosine",
            "learning_rate_warmup_steps": 2,
            "learning_rate_decay_iters": 10,
        },
        "embedding_learning_rate_scheduler": {
            "learning_rate": 0.01,
            "learning_rate_minimum": 0.0,
            "learning_rate_decay_style": "cosine",
            "learning_rate_warmup_steps": 2,
            "learning_rate_decay_iters": 10,
        },
        "training": {
            "use_separate_lr_on_embeddings": False,
        },
        "trainer": {
            "save_dir": str(tmp_path),
            "save_interval": 6,
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
            "delete_past_optimizer_states": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": {
            "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / "data"] * 10,
            "legacy_dataset": False,
            "blended_dataset": {"cache_directory": tmp_path},
        },
        "transformer_architecture": {
            "vocab_size": 128000,
            "sequence_length": 512,
            "hidden_size": 64,
            "num_attention_heads": 16,
            "num_local_attention_heads": num_local_attention_heads,
            "local_attention_window_size": local_attention_window_size,
            "num_layers": 4,
            "precision": "bfloat16",
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "masked_softmax": {"kernel": "flash_attention"},
            "causal": True,
            "norm_type": "layernorm",
            "relative_position_embedding_type": "rotary",
            "mlp_type": "default",
            "mlp_factor": 4,
            "attention_bias": True,
            "mlp_bias": True,
        },
    }

    config = TransformerConfig.from_dict(config_dict)

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume and one that was
    # trained continuously
    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        checkpoint_dir=tmp_path,
        _world_size=world_size,
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    determined_checkpoint_dir = str(Path(tmp_path) / "determined_checkpoint")
    os.environ["DET_LATEST_CHECKPOINT"] = str(determined_checkpoint_dir)

    config_dict["trainer"]["load_dir"] = str(tmp_path)
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_loaded = TransformerConfig.from_dict(config_dict)
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
        checkpoint_dir=tmp_path,
        _world_size=world_size,
    )

    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["metrics"][-4:],
        return_dict_resumed_trained_model["metrics"],
    ):
        diff_pct = abs(loss_original["training/loss"] - loss_resumed["training/loss"]) / loss_original["training/loss"]

        assert diff_pct < 1e-10, (
            f"loss changed after continuing training from a checkpoint; "
            f"expected {return_dict_continuously_trained_model['metrics'][-4:]}, "
            f"got {return_dict_resumed_trained_model['metrics']}; diff_pct {diff_pct}"
        )

    log_dir = tmp_path / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        assert (date_log_dir / "profile.json").is_file(), "did not save profile information"
