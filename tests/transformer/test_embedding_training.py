import os
from pathlib import Path
from typing import Dict

import pytest
import torch

from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from scaling.transformer.data.embedding_dataset import EmbeddingDataset
from scaling.transformer.tokenizer import Tokenizer
from scaling.transformer.train import main
from tests.core.utils import dist_launcher


def run_test_embedding_finetuning(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    losses = main(launch_config=launch_config, overwrite_config=config_dict, return_metrics=True)
    return_dict["losses"] = losses


def mem_map_exists(load_path):
    return (
        os.path.exists(load_path.with_suffix(".bin"))
        and os.path.exists(load_path.with_suffix(".idx"))
        and os.path.exists(load_path.with_suffix(".meta.json"))
        and os.path.exists(load_path.with_suffix(".done"))
    )


def construct_basic_training_config(
    cache_dir: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    use_instructed_dataset: bool,
    embedding_dataset_memory_map: bool = False,
    masked_softmax: dict = {"kernel": "flash_attention"},
):
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )
    parent = Path(__file__).parents[0]
    tokenizer = Tokenizer.from_file(str(parent / "files/unigram_02pct_cc_v1.0_hf_converted_cleaned.json"))
    if embedding_dataset_memory_map:
        if use_instructed_dataset:
            load_path = parent / "files/dataset/embedding_dataset_instructed.jsonl"
        else:
            load_path = parent / "files/dataset/embedding_dataset_non_instructed.jsonl"

        if not mem_map_exists(cache_dir / load_path.stem):
            EmbeddingDataset.jsonl_to_embedding_mmap(
                load_path, tokenizer=tokenizer, out_prefix_path=cache_dir / load_path.stem
            )
        load_path = cache_dir / load_path.stem
        data_config = {
            "data_prefixes": [load_path],
            "embedding_dataset": True,
            "embedding_dataset_memory_map": True,
        }
    else:
        if use_instructed_dataset:
            filename = "embedding_dataset_instructed.jsonl"
        else:
            filename = "embedding_dataset_non_instructed.jsonl"
        data_config = {
            "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / filename],
            "blended_dataset": {"cache_directory": cache_dir},
            "embedding_dataset": True,
            "embedding_dataset_memory_map": False,
        }

    config_dict: Dict = {
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": 1,
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
        "training_groups": [
            {
                "group_name": "param_group",
                "learning_rate_scheduler": {
                    "learning_rate": 0.1,
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
        "training": {
            "loss_function_config": {"loss_type": "contrastive_loss"},
        },
        "logger": {"log_level": "debug", "log_dir": str(cache_dir / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": data_config,
        "transformer_architecture": {
            "image_encoder": False,
            "weight_tying": False,
            "vocab_size": 128000,
            "vocab_file": Path(__file__).parents[0] / "files" / "alpha-001-128k.json",
            "sequence_length": 256,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 24,
            "precision": "bfloat16",
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "lm_head": False,
            "masked_softmax": masked_softmax,
            "embedding_head_config": {
                "name": "test",
            },
        },
    }
    return config_dict


@pytest.mark.embedding
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size,world_size",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
    ],
)
@pytest.mark.parametrize("micro_batch_size", [2])
@pytest.mark.parametrize("use_instructions", [True, False])
@pytest.mark.parametrize("query_side_only", [True, False])
@pytest.mark.parametrize("proj_layers", [None, [1024], [128, 256, 512]])
@pytest.mark.parametrize("use_memory_map", [True, False])
@pytest.mark.parametrize("use_instructed_dataset", [True, False])
def test_embedding_finetuning(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    use_instructions: bool,
    query_side_only: bool,
    proj_layers: list[int] | None,
    use_memory_map: bool,
    use_instructed_dataset: bool,
):
    """
    End-to-end test spanning the full embedding training life cycle.

    Includes:
        - Setup of model in a distributed environment
        - Training
        - Checkpointing
        - Checkpoint resume
    """

    if query_side_only and not use_instructions:
        pytest.skip("Can't run query_side_only without setting use_instruction.")

    if use_instructions and not use_instructed_dataset:
        pytest.skip("Can't run instructed run without using a instructed dataset.")

    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size,
        pipe_parallel_size,
        world_size,
        micro_batch_size,
        embedding_dataset_memory_map=use_memory_map,
        use_instructed_dataset=use_instructed_dataset,
    )

    config_dict["training"]["loss_function_config"]["use_instructions"] = use_instructions
    config_dict["training"]["loss_function_config"]["query_side_only"] = query_side_only
    config_dict["transformer_architecture"]["embedding_head_config"]["name"] = "test"
    config_dict["transformer_architecture"]["embedding_head_config"]["proj_layers"] = proj_layers

    config = TransformerConfig.from_dict(config_dict)

    _ = dist_launcher(
        run_func=run_test_embedding_finetuning,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
    )
