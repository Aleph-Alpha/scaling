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
        (2, 1, 2),
        (2, 2, 4),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16"), (False, "bfloat16")])
@pytest.mark.parametrize("weight_tying", [False])
def test_sequence_parallel_training(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    weight_tying: bool,
    legacy_dataset: bool = False,
    use_determined: bool = False,
    use_separate_lr_on_embeddings: bool = False,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
):
    """
    End-to-end test for training with and without sequence parallel and expecting the same loss

    Includes:
        - Setup of model in a distributed environment
        - Training without Sequence Parallel
        - Training with Sequence Parallel
        - Compare losses
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "runner": {"use_determined": use_determined},
        "topology": {
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "activation_checkpointing_type": "disabled",
            "sequence_parallel": False,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {
                "enable": enable_loss_scaling,
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
            "use_separate_lr_on_embeddings": use_separate_lr_on_embeddings,
        },
        "trainer": {
            "save_dir": str(tmp_path),
            "save_interval": 6,
            "load_dir": str(tmp_path),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": {
            "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / "legacy" / "enron_text_document_100"]
            * 2
            if legacy_dataset
            else [Path(__file__).parents[0] / "files" / "dataset" / "data"],
            "legacy_dataset": legacy_dataset,
            "blended_dataset": {"cache_directory": tmp_path},
        },
        "transformer_architecture": {
            "vocab_size": 128000,
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 4,
            "precision": precision,
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "masked_softmax": {"kernel": "torch"},
            "causal": True,
            "norm_type": norm_type,
            "relative_position_embedding_type": relative_position_embedding_type,
            "mlp_type": mlp_type,
            "mlp_factor": mlp_factor,
            "attention_bias": attention_bias,
            "mlp_bias": mlp_bias,
            "key_query_norm": key_query_norm,
            "weight_tying": weight_tying,
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
    if use_determined:
        determined_checkpoint_dir = str(Path(tmp_path) / "determined_checkpoint")
        os.environ["DET_LATEST_CHECKPOINT"] = str(determined_checkpoint_dir)

    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    # Switch to use sequence parallelism in second run
    config_dict["topology"]["sequence_parallel"] = True

    config_loaded = TransformerConfig.from_dict(config_dict)
    return_dict_resumed_trained_model_with_sequence_parallel = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
        checkpoint_dir=tmp_path,
        _world_size=world_size,
    )

    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["metrics"][-4:],
        return_dict_resumed_trained_model_with_sequence_parallel["metrics"],
    ):
        diff_pct = abs(loss_original["training/loss"] - loss_resumed["training/loss"]) / loss_original["training/loss"]
        assert (
            diff_pct < 1e-2  # TODO: Why is such huge loss difference needed here? This can't just be numerics...
        ), (
            f"loss changed after continuing training from a checkpoint with sequence parallelism; "
            f"expected {return_dict_continuously_trained_model['metrics'][-4:]}, "
            f"got {return_dict_resumed_trained_model_with_sequence_parallel['metrics']}; diff_pct {diff_pct}"
        )

    log_dir = tmp_path / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        assert (date_log_dir / "profile.json").is_file(), "did not save profile information"
