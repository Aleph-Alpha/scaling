import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from scaling.core.data import MemoryMapDatasetBuilder
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from scaling.transformer.tokenizer import Tokenizer
from scaling.transformer.train import main
from tests.core.utils import dist_launcher


def load_full_separated_checkpoint(checkpoint_file: Path):
    checkpoint_file_options = checkpoint_file.parent.glob(f"{checkpoint_file.stem}*.pt")

    state_dict = dict()
    for checkpoint_file_option in checkpoint_file_options:
        sd = torch.load(
            str(checkpoint_file_option),
            map_location=torch.device("cpu"),
        )
        state_dict.update(sd)

    return state_dict


def construct_basic_training_config(
    cache_dir: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    finetuning_dataset_memory_map: bool = False,
    masked_softmax: dict = {"kernel": "torch"},
    data_prefixes: list | None = None,
    weight_tying=False,
):
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    if finetuning_dataset_memory_map:
        if data_prefixes is None:
            data_prefixes = [Path(__file__).parents[0] / "files" / "dataset" / "finetuning_memory_map" / "dataset"]
        data_config = {
            "data_prefixes": data_prefixes,
            "blended_dataset": {"cache_directory": cache_dir},
            "finetuning_dataset": True,
            "finetuning_dataset_memory_map": True,
        }
    else:
        data_config = {
            "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / "finetuning.json"],
            "blended_dataset": {"cache_directory": cache_dir},
            "finetuning_dataset": True,
        }

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
            "zero": False,
        },
        "training_groups": [
            {
                "group_name": "param_group",
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
            "save_dir": str(cache_dir),
            "save_interval": 6,
            "load_dir": str(cache_dir),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "logger": {"log_level": "debug", "log_dir": str(cache_dir / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": data_config,
        "transformer_architecture": {
            "weight_tying": weight_tying,
            "vocab_size": 128000,
            "vocab_file": Path(__file__).parents[0] / "files" / "alpha-001-128k.json",
            "sequence_length": 256,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_layers": 2,
            "precision": precision,
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "masked_softmax": masked_softmax,
        },
    }
    return config_dict


def run_test_finetuning(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    losses = main(launch_config=launch_config, overwrite_config=config_dict, return_metrics=True)
    return_dict["losses"] = losses


@pytest.mark.finetuning
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 2, 4),
        (2, 2, 4),
        (2, 4, 8),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(False, "bfloat16")])
def test_transformer_softprompt_finetuning(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        gradient_accumulation_steps,
        enable_loss_scaling,
        precision,
    )

    config = TransformerConfig.from_dict(config_dict)

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    save_dir = config.trainer.save_dir
    if save_dir is None:
        raise ValueError("save_dir cannot be None.")

    save_dir_path = Path(save_dir)
    latest_global_step = (save_dir_path / "latest").read_text()
    baseline_checkpoint_path = save_dir_path / latest_global_step

    # Resume model training from the previous checkpoint at 6 steps.
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["trainer"]["load_optimizer_states"] = False
    config_dict["trainer"]["load_context"] = False
    config_dict["trainer"]["save_interval"] = 2
    config_dict["trainer"]["train_iterations"] = 4

    # allowed_missing_keys_in_checkpoint
    config_dict["trainer"]["allowed_missing_keys_in_checkpoint"] = [
        "softprompt_finetuning",
        "running_var",
        "running_mean",
    ]

    # softprompt specific
    config_dict["transformer_architecture"]["softprompt_config"] = {"name": "finetuning", "n_tokens": 8}

    config_dict["training"] = {"allow_missing_params_in_optimizer": True}
    config_dict["training_groups"] = [
        {
            "group_name": "param_group",
            "parameters_include": ["finetuning"],
            "learning_rate_scheduler": {
                "learning_rate": 0.01,
                "learning_rate_minimum": 0.0,
                "learning_rate_decay_style": "cosine",
                "learning_rate_warmup_steps": 2,
                "learning_rate_decay_iters": 10,
            },
        }
    ]

    config_loaded = TransformerConfig.from_dict(config_dict)

    # finetuning
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )

    softprompt_parameter_count = 0

    # compare new checkpoint to see that frozen weights are actually frozen and finetuning parameters finetuned
    for checkpoint_file in baseline_checkpoint_path.glob("*.pt"):
        # we are only checking parameter files
        if not checkpoint_file.name.startswith("model_state"):
            continue

        # load baseline
        baseline = torch.load(str(checkpoint_file), map_location=torch.device("cpu"))

        # load global_step2
        trained_global_step2 = load_full_separated_checkpoint(save_dir_path / "global_step2" / checkpoint_file.name)

        # load global_step4
        trained_global_step4 = load_full_separated_checkpoint(save_dir_path / "global_step4" / checkpoint_file.name)

        # count the softprompt parameters added after finetuning
        softprompt_name = config_dict["transformer_architecture"]["softprompt_config"]["name"]
        softprompt_parameters = [k for k in trained_global_step2.keys() if softprompt_name in k]
        softprompt_parameter_count += len(softprompt_parameters)

        """
        The following file has changed from baseline to finetuning:
        model_state_layer_0_TransformerEmbeddingInput.pt
        Changed parameters are:
        baseline_global_step6_layers: ['embedding.weight']
        global_step2_layers: ['softprompt_finetuning', 'embedding.weight']
        global_step4_layers: ['softprompt_finetuning', 'embedding.weight']

        Layers existing as part of "adapter finetuning"
        0:'softprompt_finetuning'
        """

        # compare finetuned parameters: global_step2 to global_step4
        # at least a parameter needs to be changed for this to pass
        for parameter in softprompt_parameters:
            assert (
                trained_global_step2[parameter] != trained_global_step4[parameter]
            ).any(), f"parameter '{parameter}' was not trained"

        # compare frozen parameters: baseline to global_step2
        # no parameter should have changed for this to pass
        for k in baseline.keys():
            assert (baseline[k] == trained_global_step2[k]).all(), f"parameter '{k}' was trained"

    # check if there have been any finetuned parameters compared
    assert softprompt_parameter_count > 0, "there have been no softprompt finetuned parameters found"


@pytest.mark.finetuning
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 2, 4),
        (2, 2, 4),
        (2, 4, 8),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(False, "bfloat16")])
@pytest.mark.parametrize("masked_softmax", [{"kernel": "flash_attention"}, {"kernel": "torch"}])
def test_transformer_adapter_finetuning(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    masked_softmax: dict,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        gradient_accumulation_steps,
        enable_loss_scaling,
        precision,
        masked_softmax=masked_softmax,
    )

    config = TransformerConfig.from_dict(config_dict)

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    save_dir = config.trainer.save_dir
    if save_dir is None:
        raise ValueError("save_dir cannot be None.")

    save_dir_path = Path(save_dir)
    latest_global_step = (save_dir_path / "latest").read_text()
    baseline_checkpoint_path = save_dir_path / latest_global_step

    # Resume model training from the previous checkpoint at 6 steps.
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["trainer"]["load_optimizer_states"] = False
    config_dict["trainer"]["load_context"] = False
    config_dict["trainer"]["save_interval"] = 2
    config_dict["trainer"]["train_iterations"] = 4

    # allowed_missing_keys_in_checkpoint
    config_dict["trainer"]["allowed_missing_keys_in_checkpoint"] = [
        "attn_adapter_finetuning.dense_in.weight",
        "attn_adapter_finetuning.dense_out.weight",
        "mlp_adapter_finetuning.dense_in.weight",
        "mlp_adapter_finetuning.dense_out.weight",
        "running_var",
        "running_mean",
        "image_encoder",
    ]

    # adapter specific

    config_dict["transformer_architecture"]["adapter_config"] = {
        "name": "finetuning",
        "attention_downsampling_factor": 0.25,
        "mlp_downsampling_factor": 0.25,
        "init_std": 0.1,
    }

    config_dict["training"] = {"allow_missing_params_in_optimizer": True}
    config_dict["training_groups"] = [
        {
            "group_name": "param_group",
            "parameters_include": ["finetuning"],
            "learning_rate_scheduler": {
                "learning_rate": 0.01,
                "learning_rate_minimum": 0.0,
                "learning_rate_decay_style": "cosine",
                "learning_rate_warmup_steps": 2,
                "learning_rate_decay_iters": 10,
            },
        }
    ]

    config_loaded = TransformerConfig.from_dict(config_dict)

    # finetuning
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )

    ### compare new checkpoint to see that frozen weights are actually frozen and finetuned are changed
    for checkpoint_file in baseline_checkpoint_path.glob("*.pt"):
        # we are only checking parameter files
        if not checkpoint_file.name.startswith("model_state"):
            continue

        # load baseline
        baseline = torch.load(str(checkpoint_file), map_location=torch.device("cpu"))

        # load global_step2
        trained_global_step2 = load_full_separated_checkpoint(save_dir_path / "global_step2" / checkpoint_file.name)

        # load global_step4
        trained_global_step4 = load_full_separated_checkpoint(save_dir_path / "global_step4" / checkpoint_file.name)

        adapter_name = config_dict["transformer_architecture"]["adapter_config"]["name"]

        if "TransformerLayer" in str(checkpoint_file):
            adapter_parameters = [k for k in trained_global_step2.keys() if adapter_name in k]

            """
            Layers existing as part of "adapter finetuning"
            0:'attn_adapter_finetuning.dense_in.weight'
            1:'attn_adapter_finetuning.dense_out.weight'
            2:'mlp_adapter_finetuning.dense_in.weight'
            3:'mlp_adapter_finetuning.dense_out.weight'
            """

            # this assertion is not considering adding adapters to certain layers only
            assert len(adapter_parameters) > 0, f"adapter not found in layer {checkpoint_file}"

            # compare finetuned parameters: global_step2 to global_step4
            # at least a parameter needs to be changed for this to pass
            for parameter in adapter_parameters:
                assert (
                    trained_global_step2[parameter] != trained_global_step4[parameter]
                ).any(), f"parameter '{parameter}' was not trained"

        # compare frozen parameters: baseline to global_step2
        # no parameter should have changed for this to pass
        for k in baseline.keys():
            assert (baseline[k] == trained_global_step2[k]).all(), f"parameter '{k}' was trained"


@pytest.mark.finetuning
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 2, 4),
        (2, 2, 4),
        (2, 4, 8),
    ],
)
@pytest.mark.parametrize("micro_batch_size,gradient_accumulation_steps", [(2, 1)])
@pytest.mark.parametrize("enable_loss_scaling,precision", [(False, "bfloat16")])
@pytest.mark.parametrize(
    "finetuning_dataset_memory_map",
    [False, True],
)
def test_transformer_bitfit_finetuning(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    enable_loss_scaling: bool,
    precision: str,
    finetuning_dataset_memory_map: bool,
):
    """
    End-to-end test spanning the full training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        gradient_accumulation_steps,
        enable_loss_scaling,
        precision,
        finetuning_dataset_memory_map=finetuning_dataset_memory_map,
    )

    config = TransformerConfig.from_dict(config_dict)

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    save_dir = config.trainer.save_dir
    if save_dir is None:
        raise ValueError("save_dir cannot be None.")

    save_dir_path = Path(save_dir)
    latest_global_step = (save_dir_path / "latest").read_text()
    baseline_checkpoint_path = save_dir_path / latest_global_step

    # Resume model training from the previous checkpoint at 6 steps.
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["trainer"]["load_optimizer_states"] = False
    config_dict["trainer"]["load_context"] = False
    config_dict["trainer"]["save_interval"] = 2
    config_dict["trainer"]["train_iterations"] = 4

    # bitfit specific
    config_dict["transformer_architecture"]["bitfit_bias_config"] = {"name": "finetuning"}

    config_dict["training"] = {"allow_missing_params_in_optimizer": True}
    config_dict["training_groups"] = [
        {
            "group_name": "param_group",
            "parameters_include": ["finetuning"],
            "learning_rate_scheduler": {
                "learning_rate": 0.01,
                "learning_rate_minimum": 0.0,
                "learning_rate_decay_style": "cosine",
                "learning_rate_warmup_steps": 2,
                "learning_rate_decay_iters": 10,
            },
        }
    ]

    # allowed_missing_keys_in_checkpoint
    config_dict["trainer"]["allowed_missing_keys_in_checkpoint"] = [
        "softprompt_finetuning",
        "running_var",
        "running_mean",
        "finetuning",
    ]
    config_dict["trainer"]["allowed_unexpected_keys_in_checkpoint"] = ["bias"]

    config_loaded = TransformerConfig.from_dict(config_dict)

    # finetuning
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )

    ### compare new checkpoint to see that frozen weights are actually frozen and biases are changed
    for checkpoint_file in baseline_checkpoint_path.glob("*.pt"):
        # we are only checking parameter files
        if not checkpoint_file.name.startswith("model_state"):
            continue

        # load baseline
        baseline = torch.load(str(checkpoint_file), map_location=torch.device("cpu"))

        # load global_step2
        trained_global_step2 = load_full_separated_checkpoint(save_dir_path / "global_step2" / checkpoint_file.name)

        if "TransformerLayer" in str(checkpoint_file):
            # compare finetuned layers
            trained_global_step4 = load_full_separated_checkpoint(save_dir_path / "global_step4" / checkpoint_file.name)

            # count the bias parameters added after finetuning
            bias_name = config_dict["transformer_architecture"]["bitfit_bias_config"]["name"]
            bias_parameters = [k for k in trained_global_step2.keys() if bias_name in k]
            assert len(bias_parameters) > 0, f"bias not found in layer {checkpoint_file}"

            """
            Layers existing as part of "bitfit finetuning"
            0:'input_layernorm.bias_finetuning'
            1:'self_attention.query_key_value.bias_finetuning'
            2:'self_attention.dense.bias_finetuning'
            3:'post_attention_layernorm.bias_finetuning'
            4:'mlp.dense_in.bias_finetuning'
            5:'mlp.dense_out.bias_finetuning'
            """

            # compare finetuned parameters: global_step2 to global_step4
            # at least a parameter needs to be changed for this to pass
            for parameter in bias_parameters:
                assert (
                    trained_global_step2[parameter] != trained_global_step4[parameter]
                ).any(), f"parameter '{parameter}' was not trained"

        # compare frozen parameters: baseline to global_step2
        # no parameter should have changed for this to pass
        for k in baseline.keys():
            # all .bias were dropped from model checkpoint to avoid confusion
            if k.endswith(".bias"):
                continue
            assert (baseline[k] == trained_global_step2[k]).all(), f"parameter '{k}' was trained"


@pytest.mark.finetuning
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 4),
    ],
)
def test_transformer_finetuning_with_ignore_keys_in_checkpoint(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
):
    """
    Tests ignore_keys_in_checkpoint option when loading checkpoint.
    This test just makes sure that everything runs, the logic of ignoring keys is tested in scaling.
    """
    # initial training
    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        enable_loss_scaling=False,
        precision="bfloat16",
    )
    config = TransformerConfig.from_dict(config_dict)
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    # finetuning with some keys being ignored in checkpoint
    ignore_keys_in_checkpoint = ["embedding.weight"]
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["trainer"]["load_optimizer_states"] = False
    config_dict["trainer"]["load_context"] = False
    config_dict["trainer"]["save_interval"] = 2
    config_dict["trainer"]["train_iterations"] = 4
    config_dict["training"] = {"allow_missing_params_in_optimizer": True}
    config_dict["training_groups"] = [
        {
            "group_name": "param_group",
            "parameters_include": ignore_keys_in_checkpoint,
            "learning_rate_scheduler": {
                "learning_rate": 0.01,
                "learning_rate_minimum": 0.0,
                "learning_rate_decay_style": "cosine",
                "learning_rate_warmup_steps": 2,
                "learning_rate_decay_iters": 10,
            },
        }
    ]

    config_loaded = TransformerConfig.from_dict(config_dict)
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )


@pytest.mark.finetuning
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
    ],
)
@pytest.mark.parametrize(
    "parameters_include",
    [
        {"embedding.weight"},
        {"embedding.weight", "mlp.dense_in.weight"},
    ],
)
@pytest.mark.parametrize(
    "weight_tying,finetunable_param,finetunable_layer",
    [(False, "linear.weight", "TransformerLMHead"), (True, "embedding.weight", "TransformerLMHeadTied")],
)
def test_transformer_finetune_embedding(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    parameters_include: set[str],
    finetunable_param: str,
    finetunable_layer: str,
    weight_tying: bool,
):
    parameters_include = parameters_include | {finetunable_param}
    prefix_path_memory_map = tmp_path / "fine_tuned_embedding" / "enriched_training_tokens"
    tokenizer_path = Path(__file__).parents[0] / "files" / "alpha-001-128k.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    builder = MemoryMapDatasetBuilder(prefix_path=Path(prefix_path_memory_map))
    ids = tokenizer.encode("Abra kadabra zweimal schwarzer Kater") + [tokenizer.eos_token_id]

    finetunable_token_ids = [0, 77, 222, 127995]

    ids += [i for i in range(4)] + [i + 64000 for i in range(4)] + finetunable_token_ids + [tokenizer.eos_token_id]
    for _ in range(10):
        builder.add(np_array=np.array(ids))
    builder.finalize()

    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size=model_parallel_size,
        pipe_parallel_size=pipe_parallel_size,
        world_size=world_size,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        enable_loss_scaling=False,
        precision="bfloat16",
        finetuning_dataset_memory_map=True,
        weight_tying=weight_tying,
    )

    config_dict["trainer"]["assert_checkpoint_loaded"] = False
    config_dict["training"] = {"allow_missing_params_in_optimizer": False}
    config_dict["data"]["data_prefixes"] = [prefix_path_memory_map]

    config = TransformerConfig.from_dict(config_dict)

    # finetuning
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )

    save_dir = config.trainer.save_dir
    if save_dir is None:
        raise ValueError("save_dir cannot be None.")

    save_dir_path = Path(save_dir)
    latest_global_step = (save_dir_path / "latest").read_text()
    baseline_checkpoint_path = save_dir_path / latest_global_step

    # Resume model training from the previous checkpoint at 6 steps.
    config_dict["trainer"]["save_interval"] = 2
    config_dict["trainer"]["train_iterations"] = 4
    config_dict["trainer"]["assert_checkpoint_loaded"] = True
    config_dict["trainer"]["load_optimizer_states"] = False
    config_dict["trainer"]["load_context"] = False
    config_dict["training"] = {"allow_missing_params_in_optimizer": True}
    config_dict["transformer_architecture"]["finetunable_token_ids"] = finetunable_token_ids

    config_dict["training_groups"] = [
        {
            "group_name": "param_group",
            "parameters_include": parameters_include,
            "learning_rate_scheduler": {
                "learning_rate": 0.01,
                "learning_rate_minimum": 0.0,
                "learning_rate_decay_style": "cosine",
                "learning_rate_warmup_steps": 2,
                "learning_rate_decay_iters": 10,
            },
        }
    ]

    config_loaded = TransformerConfig.from_dict(config_dict)
    _ = dist_launcher(
        run_func=run_test_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )
    assert_reached = [False, False]
    layers_to_check = {"EmbeddingInput", finetunable_layer}
    layers_checked = set()
    token_ids_checked = defaultdict(set)
    ### compare new checkpoint to see that frozen weights are actually frozen and biases are changed
    for checkpoint_file in baseline_checkpoint_path.glob("*.pt"):
        if not checkpoint_file.name.startswith("model_state"):
            continue
        trained_global_step2 = load_full_separated_checkpoint(save_dir_path / "global_step2" / checkpoint_file.name)
        current_layer = next((x for x in layers_to_check if x in str(checkpoint_file)), None)
        if current_layer is not None:
            layers_checked.add(current_layer)
            trained_global_step4 = load_full_separated_checkpoint(save_dir_path / "global_step4" / checkpoint_file.name)
            broken_ids = set()
            for finetunable_parameter in parameters_include:
                if finetunable_parameter in trained_global_step2:
                    for token_id, (old_embedding, new_embedding) in enumerate(
                        zip(trained_global_step2[finetunable_parameter], trained_global_step4[finetunable_parameter])
                    ):
                        if token_id in finetunable_token_ids:
                            token_ids_checked[current_layer].add(token_id)
                            assert_reached[0] = True
                            assert not torch.all(old_embedding == new_embedding)
                        else:
                            assert_reached[1] = True
                            if not torch.all(old_embedding == new_embedding):
                                broken_ids.add(token_id)
            assert broken_ids == set()
    assert all(assert_reached)
    assert layers_checked == layers_to_check
    assert all(token_ids_checked[layer] == set(finetunable_token_ids) for layer in layers_to_check)

    shutil.rmtree(tmp_path)
