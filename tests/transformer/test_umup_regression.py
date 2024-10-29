import json
from pathlib import Path
from typing import Dict

import pytest

from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from scaling.transformer.train import main
from tests.core.utils import dist_launcher


def run_test_training(
    return_dict: dict,
    config_dict: dict,
    checkpoint_dir: str,
    _world_size: int,
):
    """
    function implementing the behavior of training for one single gpu / process
    """

    launch_config = LaunchConfig.from_launcher_args()
    metrics_list = main(
        launch_config=launch_config,
        overwrite_config=config_dict,
        return_metrics=True,
    )

    if config_dict["topology"]["global_rank"] == 0:
        assert metrics_list is not None
        return_dict["metrics"] = metrics_list


def execute_run_training(
    cache_dir: Path,
):
    config_dict: Dict = {
        "runner": {"use_determined": False},
        "topology": {
            "world_size": 2,
            "model_parallel_size": 1,
            "pipe_parallel_size": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing_type": "disabled",
            "sequence_parallel": False,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 0.0,
            "loss_scaler": {
                "enable": False,
                "initial_scale": 16,
            },
            "zero": True,
        },
        "training": {"allow_missing_params_in_optimizer": True},
        "training_groups": [
            {
                "group_name": "param_group",
                "weight_decay": 0.01,
                "independent_weight_decay": True,
                "parameters_exclude": ["norm"],
                "learning_rate_scheduler": {
                    "learning_rate": 1.0,
                    "learning_rate_minimum": 1.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 0,
                    "learning_rate_decay_iters": 100,
                },
            }
        ],
        "trainer": {
            "seed": 42,
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(cache_dir / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": {
            "data_prefixes": ([Path(__file__).parents[0] / "files" / "dataset" / "legacy" / "enron_text_document_100"]),
            "legacy_dataset": True,
            "blended_dataset": {"cache_directory": cache_dir},
        },
        "transformer_architecture": {
            "weight_tying": False,
            "vocab_size": 65536,
            "hidden_size": 256,
            "num_layers": 2,
            "num_attention_heads": 2,
            "rotary_embedding_base": 10000,
            "sequence_length": 2048,
            "norm_type": "rms",
            "relative_position_embedding_type": "rotary_complex",
            "mlp_type": "swiglu",
            "mlp_factor": 2.6640625,
            "attention_bias": False,
            "mlp_bias": False,
            "masked_softmax": {"kernel": "torch", "softmax_in_fp32": True, "scale": 1.0},
            "layernorm": {"optimization_type": "torch", "layernorm_epsilon": 1.0e-20},
            "precision": "float32",
            "dropout_embedding": 0.0,
            "dropout_attention_probs": 0.0,
            "dropout_after_attention": 0.0,
            "dropout_after_mlp": 0.0,
            "umup": {
                "enable": True,
                "normalize_depth_to_num_layers": True,
                "residual_mult": 2.0,
                "residual_attn_ratio": 0.25,
                "attn_mult": 4.0,
                "act_mult": 3.0,
                "loss_mult": 0.5,
            },
            "reset_attention_mask": False,
            "reset_position_ids": False,
        },
    }
    config = TransformerConfig.from_dict(config_dict)

    return_dict = dist_launcher(
        run_func=run_test_training,
        world_size=2,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        checkpoint_dir=cache_dir,
        _world_size=2,
    )

    return return_dict["metrics"]


@pytest.mark.skipif(True, reason="umup regression test only works on matching hardware and env, see docstring")
def test_umup_regression(tmp_path: Path):
    """Checks if the losses produced by u-mup in this codebase
    match losses of the other u-mup codebase on an equivalent config

    IMPORTANT: Hardware and other environment factors influcence the expected error,
    hence why we do not run this test in the CI pipeline.

    Old losses were recorded on RTX 4090 and PyTorch 2.3.1. This test should pass on RTX 4090
    with PyTorch 2.4.0+cu121.
    """

    losses = execute_run_training(tmp_path)

    with open(Path(__file__).parent / "files" / "old_umup_losses_4090_torch_2_3_1.json", "r") as f:
        losses_old = json.load(f)

    for step, (loss_dict_old, loss_dict_new) in enumerate(zip(losses_old, losses)):
        abs_loss_diff = abs(loss_dict_old["training/loss"] - loss_dict_new["training/loss"])
        rel_loss_diff = abs_loss_diff / loss_dict_old["training/loss"]
        assert abs_loss_diff < 1.0e-5, f"absolute loss difference {abs_loss_diff} at step {step} exceeds tolerance"
        assert rel_loss_diff < 1.0e-6, f"relative loss difference {rel_loss_diff} at step {step} exceeds tolerance"
