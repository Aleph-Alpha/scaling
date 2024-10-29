import os
from pathlib import Path
from typing import Dict
from unittest import mock

import pytest
import torch

from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from scaling.transformer.train import main
from scaling.transformer.train_determined import main as main_determined
from tests.core.utils import dist_launcher, dist_launcher_runtime_error

from .utils_determined import get_determined_context, make_mock_cluster_info


@mock.patch("uuid.uuid4")
@mock.patch("determined.get_cluster_info")
def run_test_training(
    mock_cluster_info,
    mock_uuid,
    return_dict: dict,
    config_dict: dict,
    checkpoint_dir: str,
    _world_size: int,
):
    """
    function implementing the behavior of training for one single gpu / process
    """
    if config_dict["runner"]["use_determined"]:
        cluster_info = make_mock_cluster_info(_world_size, checkpoint_dir)
        cluster_info._latest_checkpoint = os.environ.get("DET_LATEST_CHECKPOINT")
        mock_cluster_info.return_value = cluster_info
        mock_uuid.return_value = "determined_checkpoint"
        with get_determined_context(checkpoint_dir) as determined_context:
            metrics_list = main_determined(
                determined_context,
                None,
                overwrite_config=config_dict,
                return_metrics=True,
                info=cluster_info,
            )
    else:
        launch_config = LaunchConfig.from_launcher_args()
        metrics_list = main(
            launch_config=launch_config,
            overwrite_config=config_dict,
            return_metrics=True,
        )

    if config_dict["topology"]["global_rank"] == 0:
        assert metrics_list is not None
        return_dict["metrics"] = metrics_list


@pytest.mark.transformer_training
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
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16"), (False, "float32")])
@pytest.mark.parametrize("legacy_dataset", [False])
@pytest.mark.parametrize("use_determined", [False])
@pytest.mark.parametrize("weight_tying,use_umup", [[True, False], [False, True]])
@pytest.mark.parametrize("kernel", ["torch", "flash_attention"])
@pytest.mark.parametrize(
    "sequence_parallel", [False]
)  # TODO: Address sequence parallel resuming numerical differences later and merge for now.
def test_transformer_training(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    legacy_dataset: bool,
    use_determined: bool,
    weight_tying: bool,
    use_umup: bool,
    kernel: str,
    sequence_parallel: bool,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """
    execute_run_training(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        gradient_accumulation_steps,
        enable_loss_scaling,
        precision,
        legacy_dataset,
        use_determined,
        weight_tying,
        use_umup,
        kernel,
        sequence_parallel,
        norm_type=norm_type,
        relative_position_embedding_type=relative_position_embedding_type,
        mlp_type=mlp_type,
        mlp_factor=mlp_factor,
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        key_query_norm=key_query_norm,
    )


def execute_run_training(
    cache_dir: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    legacy_dataset: bool,
    use_determined: bool,
    weight_tying: bool,
    use_umup: bool,
    kernel: str,
    sequence_parallel: bool,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
    use_deterministic_torch_algorithms: bool = False,
):
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )
    if precision == "float32" and kernel == "flash_attention":
        pytest.skip("skip test because flash attention does not support float32")

    if mlp_bias or attention_bias:
        training_groups = [
            {
                "group_name": "param_group",
                "parameters_exclude": [".bias"],
                "weight_decay": 0.001,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
            {
                "group_name": "param_group_no_weight_decay",
                "parameters_include": [".bias"],
                "weight_decay": 0.0,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
        ]
    else:
        training_groups = [
            {
                "group_name": "param_group",
                "weight_decay": 0.001,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            }
        ]

    config_dict: Dict = {
        "runner": {"use_determined": use_determined},
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "activation_checkpointing_type": "disabled",
            "sequence_parallel": sequence_parallel,
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
        "training": {
            "use_deterministic_torch_algorithms": use_deterministic_torch_algorithms,
        },
        "training_groups": training_groups,
        "trainer": {
            "save_dir": str(cache_dir),
            "save_interval": 6,
            "load_dir": str(cache_dir),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(cache_dir / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": {
            "data_prefixes": (
                [Path(__file__).parents[0] / "files" / "dataset" / "legacy" / "enron_text_document_100"] * 2
                if legacy_dataset
                else [Path(__file__).parents[0] / "files" / "dataset" / "data"]
            ),
            "legacy_dataset": legacy_dataset,
            "blended_dataset": {"cache_directory": cache_dir},
        },
        "transformer_architecture": {
            "vocab_size": 128000,
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 2,
            "precision": precision,
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "masked_softmax": {"kernel": kernel},
            "causal": True,
            "norm_type": norm_type,
            "relative_position_embedding_type": relative_position_embedding_type,
            "mlp_type": mlp_type,
            "mlp_factor": mlp_factor,
            "attention_bias": attention_bias,
            "mlp_bias": mlp_bias,
            "key_query_norm": key_query_norm,
            "weight_tying": weight_tying,
            "umup": {
                "enable": use_umup,
                "attn_mult": 2.0,
                "act_mult": 2.0,
                "residual_mult": 3.0,
                "residual_attn_ratio": 0.1,
                "loss_mult": 0.5,
            },
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
        checkpoint_dir=cache_dir,
        _world_size=world_size,
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    if use_determined:
        determined_checkpoint_dir = str(Path(cache_dir) / "determined_checkpoint")
        os.environ["DET_LATEST_CHECKPOINT"] = str(determined_checkpoint_dir)

    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_loaded = TransformerConfig.from_dict(config_dict)
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
        checkpoint_dir=cache_dir,
        _world_size=world_size,
    )

    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["metrics"][-4:],
        return_dict_resumed_trained_model["metrics"],
    ):
        diff_pct = abs(loss_original["training/loss"] - loss_resumed["training/loss"]) / loss_original["training/loss"]
        if sequence_parallel:
            assert diff_pct < 1e-1, (
                f"loss changed after continuing training from a checkpoint; "
                f"expected {return_dict_continuously_trained_model['metrics'][-4:]}, "
                f"got {return_dict_resumed_trained_model['metrics']}; diff_pct {diff_pct}"
            )
        else:
            assert diff_pct < 1e-10, (
                f"loss changed after continuing training from a checkpoint; "
                f"expected {return_dict_continuously_trained_model['metrics'][-4:]}, "
                f"got {return_dict_resumed_trained_model['metrics']}; diff_pct {diff_pct}"
            )

    log_dir = cache_dir / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        assert (date_log_dir / "profile.json").is_file(), "did not save profile information"


@pytest.mark.frozen_image_encoder
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16")])
@pytest.mark.parametrize("legacy_dataset", [True])
@pytest.mark.parametrize("use_determined", [False])
@pytest.mark.parametrize("weight_tying", [True])
def test_train_frozen_image_encoder(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    legacy_dataset: bool,
    use_determined: bool,
    weight_tying: bool,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
):
    """
    The image encoder contains not only parameters but also buffers.
    These need to be saved.
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "runner": {"use_determined": use_determined},
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
                "enable": enable_loss_scaling,
                "initial_scale": 16,  # set low initial loss scale to actually perform a train step in this short test
            },
            "zero": True,
        },
        "training": {
            "allow_missing_params_in_optimizer": True,
        },
        "training_groups": [
            {
                "group_name": "param_group",
                "parameters_exclude": [".bias", "image_encoder"],
                "weight_decay": 0.001,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
            {
                "group_name": "param_group_no_weight_decay",
                "parameters_include": [".bias"],
                "parameters_exclude": ["image_encoder"],
                "weight_decay": 0.0,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
        ],
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
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 2,
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
            "image_encoder": True,
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


@pytest.mark.training
def test_transformer_example_training_determined(tmp_path: Path):
    test_transformer_training(
        tmp_path=tmp_path,
        model_parallel_size=1,
        pipe_parallel_size=1,
        sequence_parallel=False,
        world_size=1,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        enable_loss_scaling=False,
        precision="bfloat16",
        legacy_dataset=False,
        use_determined=True,
        weight_tying=False,
        use_umup=False,
        kernel="torch",
    )


@pytest.mark.training
def test_transformer_example_training_component_selection(tmp_path: Path):
    test_transformer_training(
        tmp_path=tmp_path,
        model_parallel_size=1,
        pipe_parallel_size=1,
        world_size=1,
        micro_batch_size=2,
        sequence_parallel=False,
        gradient_accumulation_steps=1,
        enable_loss_scaling=True,
        precision="float16",
        legacy_dataset=True,
        use_determined=False,
        norm_type="rms",
        relative_position_embedding_type="rotary_complex",
        mlp_type="swiglu",
        mlp_factor=2,
        attention_bias=False,
        mlp_bias=False,
        key_query_norm=True,
        weight_tying=False,
        use_umup=False,
        kernel="torch",
    )


@pytest.mark.training
def test_cannot_train_without_any_trainable_parameter(
    tmp_path: Path,
    model_parallel_size: int = 1,
    pipe_parallel_size: int = 1,
    world_size: int = 1,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    enable_loss_scaling: bool = False,
    precision: str = "bfloat16",
    legacy_dataset: bool = True,
    use_determined: bool = False,
    weight_tying: bool = True,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "runner": {"use_determined": use_determined},
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
            "zero": True,
        },
        "training_groups": [
            {
                "group_name": "param_group",
                "parameters_exclude": [".bias"],
                "weight_decay": 0.00,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
            {
                "group_name": "param_group_no_weight_decay",
                "parameters_include": [".bias"],
                "weight_decay": 0.0,
                "learning_rate_scheduler": {
                    "learning_rate": 0.01,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
        ],
        "trainer": {
            "save_dir": str(tmp_path),
            "save_interval": 1,
            "load_dir": str(tmp_path),
            "train_iterations": 1,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
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
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 2,
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

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume and one that was
    # trained continuously
    config = TransformerConfig.from_dict(
        config_dict,
        overwrite_values={
            "training": {
                "allow_missing_params_in_optimizer": True,
            },
            "training_groups": [
                {
                    "parameters_include": ["this_does_not_exist"],
                }
            ],
            "transformer_architecture": {
                "image_encoder": True,
                "bitfit_bias_config": {"name": "symmetric"},
            },
        },
    )
    with pytest.raises(RuntimeError):
        _ = dist_launcher_runtime_error(
            run_func=run_test_training,
            world_size=world_size,
            master_port=find_free_port(),
            config_dict=config.as_dict(),
            checkpoint_dir=tmp_path,
            _world_size=world_size,
        )


@pytest.mark.training
def test_load_checkpoint_for_finetuning(
    tmp_path: Path,
    model_parallel_size: int = 1,
    pipe_parallel_size: int = 1,
    world_size: int = 1,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    enable_loss_scaling: bool = False,
    precision: str = "bfloat16",
    legacy_dataset: bool = True,
    use_determined: bool = False,
    weight_tying: bool = True,
    norm_type: str = "layernorm",
    relative_position_embedding_type: str = "rotary",
    mlp_type: str = "default",
    mlp_factor: float = 4,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    key_query_norm: bool = False,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "runner": {"use_determined": use_determined},
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
            "zero": True,
        },
        "training": {
            "allow_missing_params_in_optimizer": True,
        },
        "training_groups": [
            {
                "group_name": "param_group",
                "parameters_include": ["nice_new_bias"],
                "parameters_exclude": [],
                "learning_rate_scheduler": {
                    "learning_rate": 0.01,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            }
        ],
        "trainer": {
            "save_dir": str(tmp_path),
            "save_interval": 1,
            "load_dir": str(tmp_path),
            "train_iterations": 1,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
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
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 2,
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

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume and one that was
    # trained continuously
    config = TransformerConfig.from_dict(
        config_dict,
        overwrite_values={
            "training": {
                "allow_missing_params_in_optimizer": True,
            },
            "training_groups": [
                {
                    "parameters_include": ["bias"],
                    "parameters_exclude": ["image_encoder", "bias_"],
                }
            ],
            "transformer_architecture": {
                "image_encoder": True,
            },
        },
    )
    _ = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        checkpoint_dir=tmp_path,
        _world_size=world_size,
    )

    # Finetune from the previous checkpoint
    if use_determined:
        determined_checkpoint_dir = str(Path(tmp_path) / "determined_checkpoint")
        os.environ["DET_LATEST_CHECKPOINT"] = str(determined_checkpoint_dir)

    config_loaded = TransformerConfig.from_dict(
        config_dict,
        overwrite_values={
            "training": {
                "allow_missing_params_in_optimizer": True,
            },
            "training_groups": [
                {
                    "parameters_include": ["nice_new_bias"],
                    "parameters_exclude": [],
                }
            ],
            "trainer": {
                "load_optimizer_states": False,
                "load_context": False,
                "assert_checkpoint_loaded": True,
                "allowed_unexpected_keys_in_checkpoint": ["bias"],
                "train_iterations": 2,
                "allowed_missing_keys_in_checkpoint": ["running_mean", "running_var", "nice_new_bias"],
                # "save_dir": None,
            },
            "transformer_architecture": {
                "image_encoder": True,
                "bitfit_bias_config": {"name": "nice_new_bias"},
            },
        },
    )

    _ = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
        checkpoint_dir=tmp_path,
        _world_size=world_size,
    )


@pytest.mark.training
@pytest.mark.parametrize("use_deterministic_torch_algorithms", [True])
def test_train_with_deterministic_behaviour(tmp_path, use_deterministic_torch_algorithms):
    execute_run_training(
        cache_dir=tmp_path,
        model_parallel_size=1,
        pipe_parallel_size=1,
        sequence_parallel=False,
        world_size=1,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        enable_loss_scaling=False,
        precision="bfloat16",
        legacy_dataset=False,
        use_determined=True,
        weight_tying=False,
        use_umup=False,
        kernel="torch",
        use_deterministic_torch_algorithms=use_deterministic_torch_algorithms,
    )


@pytest.mark.training
@pytest.mark.parametrize("pipe_parallel_size", [1, 2])
def test_train_optimizer_group_pip_parallel(tmp_path: Path, pipe_parallel_size: int):
    """
    The image encoder contains not only parameters but also buffers.
    These need to be saved.
    """

    world_size = pipe_parallel_size

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    config_dict: Dict = {
        "runner": {"use_determined": False},
        "topology": {
            "world_size": world_size,
            "model_parallel_size": 1,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
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
        "training": {
            "allow_missing_params_in_optimizer": True,
        },
        "training_groups": [
            {
                "group_name": "param_group_token_embedding",
                "parameters_include": ["embedding.weight"],
                "weight_decay": 0.001,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
            {
                "group_name": "param_group_prediction_head",
                "parameters_include": ["linear.weight"],
                "weight_decay": 0.001,
                "learning_rate_scheduler": {
                    "learning_rate": 0.001,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            },
        ],
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
            "data_prefixes": ([Path(__file__).parents[0] / "files" / "dataset" / "data"]),
            "legacy_dataset": False,
            "blended_dataset": {"cache_directory": tmp_path},
        },
        "transformer_architecture": {
            "vocab_size": 128000,
            "sequence_length": 64,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_layers": 4,
            "precision": "bfloat16",
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "masked_softmax": {"kernel": "torch"},
            "causal": True,
            "norm_type": "layernorm",
            "relative_position_embedding_type": "rotary",
            "mlp_type": "default",
            "mlp_factor": 4,
            "attention_bias": True,
            "mlp_bias": True,
            "key_query_norm": False,
            "weight_tying": False,
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
