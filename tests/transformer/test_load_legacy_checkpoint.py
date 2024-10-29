from pathlib import Path
from typing import Any

import pytest
from _pytest.tmpdir import TempPathFactory
from pytest import fixture

from scaling.core.nn.parallel_module.partitioned_module import _is_legacy_checkpoint
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context import TransformerConfig
from scaling.transformer.train import main
from tests.core.utils import dist_launcher


def construct_basic_training_config(
    cache_dir: Path,
) -> dict[str, Any]:
    config_dict = {
        "trainer": {
            "assert_checkpoint_loaded": True,
            "load_optimizer_states": False,
            "load_context": False,
            "save_interval": 2,
            "train_iterations": 3,
            "load_dir": str(Path(__file__).parents[0] / "files" / "checkpoint_legacy"),
        },
        "data": {
            "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / "finetuning_chat.jsonl"],
            "blended_dataset": {"cache_directory": cache_dir},
            "finetuning_chat_dataset": True,
        },
        "training_groups": [
            {
                "group_name": "optimizer_group",
                "learning_rate_scheduler": {
                    "learning_rate": 0.1,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            }
        ],
        "luminous_architecture": {
            "weight_tying": False,
            "vocab_size": 2048,
            "vocab_file": str(Path(__file__).parents[0] / "files" / "checkpoint_legacy" / "vocab.json"),
            "hidden_size": 2,
            "num_layers": 1,
            "num_attention_heads": 1,
            "rotary_embedding_base": 10000,
            "sequence_length": 1024,
            "norm_type": "rms",
            "relative_position_embedding_type": "none",
            "attention_bias": False,
            "mlp_type": "swiglu",
            "mlp_factor": 2.5,
            "mlp_bias": False,
            "precision": "bfloat16",
            "image_encoder": False,
        },
    }

    return config_dict


def run_training(return_dict: dict, config_dict: dict):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    losses = main(launch_config=launch_config, overwrite_config=config_dict, return_metrics=True)
    return_dict["losses"] = losses


@pytest.mark.transformer
def test_legacy_checkpoint_loading(
    tmp_path: Path,
) -> None:
    config_dict = construct_basic_training_config(
        tmp_path,
    )

    config_loaded = TransformerConfig.from_dict(config_dict)

    _ = dist_launcher(
        run_func=run_training,
        world_size=1,
        master_port=find_free_port(),
        config_dict=config_loaded.as_dict(),
    )


@fixture
def create_tmp_dirs(tmp_path_factory: TempPathFactory) -> list[Path]:
    tmp_dirs = [tmp_path_factory.mktemp(f"dir_{i}") for i in range(3)]
    return tmp_dirs


@pytest.mark.transformer
def test_is_not_legacy_checkpoint(create_tmp_dirs: list[Path]) -> None:
    assert not _is_legacy_checkpoint(create_tmp_dirs)
    (create_tmp_dirs[0] / "model_state_layer_0_EmbeddingInput.pt").touch()
    assert not _is_legacy_checkpoint(create_tmp_dirs)


@pytest.mark.transformer
def test_is_legacy_checkpoint(create_tmp_dirs: list[Path]) -> None:
    assert not _is_legacy_checkpoint(create_tmp_dirs)
    (create_tmp_dirs[0] / "model_state_layer_0_LuminousLayerIO.pt").touch()
    assert _is_legacy_checkpoint(create_tmp_dirs)
    assert not _is_legacy_checkpoint(create_tmp_dirs[1:])
