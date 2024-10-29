from pathlib import Path
from typing import Dict

import pytest
import torch

from scaling.core.utils.port import find_free_port
from tests.core.minimal import MinimalConfig, main
from tests.core.utils import dist_launcher


def run_test_training(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    metrics_list = main(overwrite_config=config_dict, return_metrics=True)

    if config_dict["topology"]["global_rank"] == 0:
        assert metrics_list is not None
        return_dict["losses"] = [metric["training/loss"] for metric in metrics_list]


def run_test_training_wrapper(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    activation_checkpointing_type: str,
    weight_tying: bool,
    bitfit_bias_name,
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
        "topology": {
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "activation_checkpointing_type": activation_checkpointing_type,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {"enable": enable_loss_scaling, "initial_scale": 2},
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
        "training": {
            "precision": precision,
            "weight_tying": weight_tying,
            "bitfit_bias_name": bitfit_bias_name,
        },
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
    }
    config = MinimalConfig.from_dict(config_dict)

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume
    # and one that was trained continuously
    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_loaded = MinimalConfig.from_dict(config_dict)
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )

    assert return_dict_continuously_trained_model["losses"][-4:] == return_dict_resumed_trained_model["losses"], (
        f"loss changed after continuing training from a checkpoint; "
        f"expected {return_dict_continuously_trained_model['losses'][-4:]}, "
        f"got {return_dict_resumed_trained_model['losses']}"
    )

    log_dir = tmp_path / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        # assert (date_log_dir / "profile.json").is_file(), "did not save profile information"

    # make sure tied weights are the same
    # the model architecture is hardcoded here
    if weight_tying:
        embedding_weights = list()
        for file in (tmp_path / "global_step6").glob("*MinimalEmbedding*"):
            state_dict = torch.load(str(file))
            embedding_weights.append(state_dict["embedding.weight"])

        assert len(embedding_weights) == 2, "did not find the expected two weight tied layers"
        assert (embedding_weights[0] == embedding_weights[1]).all(), "tied parameters are different"

    # make sure the right biases are trained
    if bitfit_bias_name is not None:
        for file in (tmp_path / "global_step6").glob("model_state_*.pt"):
            state_dict = torch.load(str(file))
            if bitfit_bias_name == "":
                bitfit_bias_name = None
            bitfit_bias_name_in_state_dict = ".bias" if bitfit_bias_name is None else f".bias_{bitfit_bias_name}"
            for k, v in state_dict.items():
                if k.endswith(bitfit_bias_name_in_state_dict):
                    assert not (v == 0.0).all()


@pytest.mark.scaling_training
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
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1), (2, 2)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16"), (False, "float32")])
@pytest.mark.parametrize("activation_checkpointing_type", ["disabled", "every_pipe_stage"])
@pytest.mark.parametrize("weight_tying", [True, False])
@pytest.mark.parametrize("bitfit_bias_name", [None])
def test_training_baseline(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    activation_checkpointing_type: str,
    weight_tying: bool,
    bitfit_bias_name: str,
):
    run_test_training_wrapper(
        tmp_path=tmp_path,
        model_parallel_size=model_parallel_size,
        pipe_parallel_size=pipe_parallel_size,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        enable_loss_scaling=enable_loss_scaling,
        precision=precision,
        activation_checkpointing_type=activation_checkpointing_type,
        weight_tying=weight_tying,
        bitfit_bias_name=bitfit_bias_name,
    )


@pytest.mark.training_extension
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
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1), (2, 2)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "bfloat16")])
@pytest.mark.parametrize("activation_checkpointing_type", ["disabled"])
@pytest.mark.parametrize("weight_tying", [True])
@pytest.mark.parametrize(
    "bitfit_bias_name",
    [None, "test", ""],
)
def test_training_extension(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    activation_checkpointing_type: str,
    weight_tying: bool,
    bitfit_bias_name,
):
    run_test_training_wrapper(
        tmp_path=tmp_path,
        model_parallel_size=model_parallel_size,
        pipe_parallel_size=pipe_parallel_size,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        enable_loss_scaling=enable_loss_scaling,
        precision=precision,
        activation_checkpointing_type=activation_checkpointing_type,
        weight_tying=weight_tying,
        bitfit_bias_name=bitfit_bias_name,
    )


@pytest.mark.resume_layout
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size,gradient_accumulation_steps,activation_checkpointing_type,zero,model_parallel_size_resume,pipe_parallel_size_resume,world_size_resume,gradient_accumulation_steps_resume,activation_checkpointing_type_resume,zero_resume,bitfit_bias_name,bitfit_bias_name_resume",
    [
        (
            # change activation checkpointing
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            1,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            False,  # zero
            1,  # model_parallel_size_resume
            1,  # pipe_parallel_size_resume
            1,  # world_size_resume
            1,  # gradient_accumulation_steps_resume
            "every_pipe_stage",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from pipeline to model parallel
            1,  # model_parallel_size
            2,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            False,  # zero
            2,  # model_parallel_size_resume
            1,  # pipe_parallel_size_resume
            2,  # world_size_resume
            1,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from data to model parallel
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            False,  # zero
            2,  # model_parallel_size_resume
            1,  # pipe_parallel_size_resume
            2,  # world_size_resume
            2,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from data to pipe parallel
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            False,  # zero
            1,  # model_parallel_size_resume
            2,  # pipe_parallel_size_resume
            2,  # world_size_resume
            2,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from zero to non-zero with data parallel
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            True,  # zero
            1,  # model_parallel_size_resume
            1,  # pipe_parallel_size_resume
            2,  # world_size_resume
            1,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from zero to non-zero with model parallel
            2,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            True,  # zero
            2,  # model_parallel_size_resume
            1,  # pipe_parallel_size_resume
            2,  # world_size_resume
            1,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from data parallel to pipe parallel with zero
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            True,  # zero
            1,  # model_parallel_size_resume
            2,  # pipe_parallel_size_resume
            2,  # world_size_resume
            2,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            True,  # zero_resume
            None,
            None,
        ),
        (
            # switch from data parallel zero to pipe parallel with non-zero
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            True,  # zero
            1,  # model_parallel_size_resume
            2,  # pipe_parallel_size_resume
            2,  # world_size_resume
            2,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            None,
            None,
        ),
        (
            # switch from data parallel zero to pipe parallel with non-zero
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            2,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
            True,  # zero
            1,  # model_parallel_size_resume
            2,  # pipe_parallel_size_resume
            2,  # world_size_resume
            2,  # gradient_accumulation_steps_resume
            "disabled",  # activation_checkpointing_type_resume
            False,  # zero_resume
            "test",
            "test",
        ),
    ],
)
@pytest.mark.parametrize("micro_batch_size", [2])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(True, "float16"), (False, "float32")])
@pytest.mark.parametrize("weight_tying", [True, False])
def test_training_resume_with_different_layout(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    activation_checkpointing_type: str,
    zero: bool,
    model_parallel_size_resume: int,
    pipe_parallel_size_resume: int,
    world_size_resume: int,
    gradient_accumulation_steps_resume: int,
    activation_checkpointing_type_resume: str,
    zero_resume: bool,
    micro_batch_size: int,
    enable_loss_scaling: bool,
    precision: str,
    weight_tying: bool,
    bitfit_bias_name: str | None,
    bitfit_bias_name_resume: str | None,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    if world_size > torch.cuda.device_count() or world_size_resume > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} / {world_size_resume} "
            f"with available {torch.cuda.device_count()} cuda devices"
        )

    # assert batch size will be the same
    global_batch_size = (
        (world_size / model_parallel_size / pipe_parallel_size) * micro_batch_size * gradient_accumulation_steps
    )
    global_batch_size_resume = (
        (world_size_resume / model_parallel_size_resume / pipe_parallel_size_resume)
        * micro_batch_size
        * gradient_accumulation_steps_resume
    )
    assert global_batch_size == global_batch_size_resume, (
        f"global batch size must stay constant for this test to work; "
        f"got {global_batch_size} vs. {global_batch_size_resume}"
    )

    config = MinimalConfig.from_dict(
        {
            "topology": {
                "model_parallel_size": model_parallel_size,
                "pipe_parallel_size": pipe_parallel_size,
                "micro_batch_size": micro_batch_size,
                "world_size": world_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "activation_checkpointing_type": activation_checkpointing_type,
            },
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.99,
                "gradient_clipping": 1.0,
                "loss_scaler": {"enable": enable_loss_scaling, "initial_scale": 2},
                "zero": zero,
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
            "training": {
                "precision": precision,
                "weight_tying": weight_tying,
                "bitfit_bias_name": bitfit_bias_name,
            },
            "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        }
    )

    config_resume = MinimalConfig.from_dict(
        {
            "topology": {
                "model_parallel_size": model_parallel_size_resume,
                "pipe_parallel_size": pipe_parallel_size_resume,
                "world_size": world_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps_resume,
                "activation_checkpointing_type": activation_checkpointing_type_resume,
            },
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.99,
                "gradient_clipping": 1.0,
                "loss_scaler": {"enable": enable_loss_scaling, "initial_scale": 2},
                "zero": zero_resume,
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
            },
            "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
            "training": {
                "precision": precision,
                "weight_tying": weight_tying,
                "bitfit_bias_name": bitfit_bias_name_resume,
            },
            "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        }
    )

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume
    # and one that was trained continuously
    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size_resume,
        master_port=find_free_port(),
        config_dict=config_resume.as_dict(),
    )

    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["losses"][-4:],
        return_dict_resumed_trained_model["losses"],
    ):
        diff_pct = abs(loss_original - loss_resumed) / loss_original
        # TODO the difference accepted is quite large to to numeric differences due to layout and a
        #  relatively untrained model; wandb test with longer training look good. Adjust test case?
        assert diff_pct < 0.15, (
            f"loss changed after continuing training from a checkpoint; "
            f"expected {return_dict_continuously_trained_model['losses'][-4:]}, "
            f"got {return_dict_resumed_trained_model['losses']}; diff_pct {diff_pct}"
        )

    log_dir = tmp_path / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        assert (date_log_dir / "profile.json").is_file(), "did not save profile information"


@pytest.mark.training_save_checkpoints
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size,gradient_accumulation_steps,activation_checkpointing_type",
    [
        (
            # change activation checkpointing
            1,  # model_parallel_size
            1,  # pipe_parallel_size
            1,  # world_size
            1,  # gradient_accumulation_steps
            "disabled",  # activation_checkpointing_type
        ),
    ],
)
@pytest.mark.parametrize("micro_batch_size", [2])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(False, "float32")])
@pytest.mark.parametrize("weight_tying", [True])
def test_training_save_different_checkpoint_files(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    activation_checkpointing_type: str,
    micro_batch_size: int,
    enable_loss_scaling: bool,
    precision: str,
    weight_tying: bool,
):
    """
    End-to-end test spanning the full training life cycle testing the save of different files for special parameters.

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

    config = MinimalConfig.from_dict(
        {
            "topology": {
                "model_parallel_size": model_parallel_size,
                "pipe_parallel_size": pipe_parallel_size,
                "world_size": world_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "activation_checkpointing_type": activation_checkpointing_type,
            },
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.99,
                "gradient_clipping": 1.0,
                "loss_scaler": {"enable": enable_loss_scaling, "initial_scale": 2},
                "zero": True,
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
                "separate_file_for_parameters": ["bias"],
            },
            "logger": {"log_level": "debug", "log_dir": str(tmp_path / "logs")},
            "training": {
                "precision": precision,
                "weight_tying": weight_tying,
                "bitfit_bias_name": None,
            },
            "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        }
    )

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume
    # and one that was trained continuously
    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    return_dict_resumed_trained_model = dist_launcher(
        run_func=run_test_training,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    for loss_original, loss_resumed in zip(
        return_dict_continuously_trained_model["losses"][-4:],
        return_dict_resumed_trained_model["losses"],
    ):
        diff_pct = abs(loss_original - loss_resumed) / loss_original
        assert diff_pct < 0.15, (
            f"loss changed after continuing training from a checkpoint; "
            f"expected {return_dict_continuously_trained_model['losses'][-4:]}, "
            f"got {return_dict_resumed_trained_model['losses']}; diff_pct {diff_pct}"
        )

    checkpoint_files = set(f.name for f in tmp_path.glob("**/model_state_layer_*.pt"))
    assert len(checkpoint_files) == 8, (
        "saves only one file per layer, expected are biases in a separate file " "for this test case"
    )
    for expected_file in [
        "model_state_layer_0_MinimalEmbeddingInput.pt",
        "model_state_layer_1_MinimalLinearColumnParallel.pt",
        "model_state_layer_1_MinimalLinearColumnParallel_bias.pt",
        "model_state_layer_2_MinimalEmbeddingTied.pt",
        "model_state_layer_3_MinimalLinearRowParallel.pt",
        "model_state_layer_3_MinimalLinearRowParallel_bias.pt",
        "model_state_layer_4_MinimalLayerNorm.pt",
        "model_state_layer_4_MinimalLayerNorm_bias.pt",
    ]:
        assert expected_file in checkpoint_files, f"did not save {expected_file}"

    log_dir = tmp_path / "logs"
    for date_log_dir in log_dir.glob("*"):
        if not date_log_dir.is_dir():
            continue
        assert (date_log_dir / "profile.json").is_file(), "did not save profile information"
