from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

from scaling.core import Topology
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.minimal import MinimalConfig, MinimalContext
from tests.core.minimal.model import init_model
from tests.core.minimal.model.model import (
    MinimalParallelModule,
)
from tests.core.utils import dist_launcher


def run_save_and_reload_parallel_module(
    return_dict: dict,
    save_dir_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    allowed_missing_keys_in_checkpoint: Optional[List[str]] = None,
    allowed_unexpected_keys_in_checkpoint: Optional[List[str]] = None,
    ignore_keys_in_checkpoint: Optional[List[str]] = None,
):
    """
    Initializes a model, saves checkpoint, reloads the checkpoint (with options) and saves another checkpoint.

    Used to test options in `load_checkpoint`.
    """
    # get configuration from launcher
    launch_config = LaunchConfig.from_launcher_args()
    config_dict: Dict[str, Any] = dict()
    config_dict["topology"] = dict()
    config_dict["topology"]["model_parallel_size"] = model_parallel_size
    config_dict["topology"]["pipe_parallel_size"] = pipe_parallel_size
    config_dict["topology"]["world_size"] = launch_config.world_size
    config_dict["topology"]["global_rank"] = launch_config.global_rank
    config_dict["topology"]["local_slot"] = launch_config.local_slot
    config_dict["topology"]["micro_batch_size"] = 2
    config_dict["topology"]["gradient_accumulation_steps"] = 1

    # initialize
    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = MinimalContext(config=config, topology=topology)
    context.initialize(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        seed=config.trainer.seed,
    )

    # initialize model and save a checkpoint
    model = init_model(context=context)
    (save_dir_path / "original").mkdir(parents=True, exist_ok=True)
    (save_dir_path / "reloaded").mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(save_dir_path / "original", separate_file_for_parameters=None)

    # reload checkpoint (with options) and save another one
    model = init_model(context=context)
    model.load_checkpoint(
        save_dir_path / "original",
        allowed_missing_keys_in_checkpoint=allowed_missing_keys_in_checkpoint,
        allowed_unexpected_keys_in_checkpoint=allowed_unexpected_keys_in_checkpoint,
        ignore_keys_in_checkpoint=ignore_keys_in_checkpoint,
    )
    model.save_checkpoint(save_dir_path / "reloaded", separate_file_for_parameters=None)


@pytest.mark.skip(reason="unclear if tests makes sense - see todo")
@pytest.mark.parametrize(
    "model_parallel_size,pipe_parallel_size",
    [(2, 1), (1, 2), (2, 2)],
)
@pytest.mark.parametrize("ignore_keys_in_checkpoint", [["embedding.weight"]])
def test_load_checkpoint_with_ignore_keys(
    tmp_path: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    ignore_keys_in_checkpoint: List[str],
):
    """Tests the ignore_keys_in_checkpoint feature in load_checkpoint."""
    world_size = model_parallel_size * pipe_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    _ = dist_launcher(
        run_func=run_save_and_reload_parallel_module,
        world_size=world_size,
        master_port=find_free_port(),
        save_dir_path=tmp_path,
        model_parallel_size=model_parallel_size,
        pipe_parallel_size=pipe_parallel_size,
        allowed_missing_keys_in_checkpoint=[
            "embedding.weight"  # todo: should the string need to be set in both allowed missing keys and ignored keys?
        ],
        ignore_keys_in_checkpoint=ignore_keys_in_checkpoint,
    )

    path_original = tmp_path / "original"
    path_reloaded = tmp_path / "reloaded"

    ### compare the two checkpoints and check whether ignored keys have been handled adequately
    for checkpoint_file in path_original.glob("*.pt"):
        # we are only checking parameter files
        if not checkpoint_file.name.startswith("model_state"):
            continue

        # original and reloaded checkpoint
        original = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        reloaded = torch.load(path_reloaded / checkpoint_file.name, map_location=torch.device("cpu"))

        # ensure ignore_keys_in_checkpoint are reinitialized and others are not
        for k in original.keys():
            if k in ignore_keys_in_checkpoint:
                assert not torch.allclose(original[k], reloaded[k])
            else:
                assert torch.equal(original[k], reloaded[k])


@pytest.fixture
def minimal_model() -> MinimalParallelModule:
    config_dict = {
        "topology": {
            "model_parallel_size": 1,
            "pipe_parallel_size": 1,
            "world_size": 1,
            "global_rank": 0,
            "local_slot": 0,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
        }
    }

    # initialize
    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = MinimalContext(config=config, topology=topology)
    context.initialize(
        master_addr="",
        master_port="",
        seed=config.trainer.seed,
        distributed=False,
    )

    # initialize model
    model = init_model(context=context)

    return model


def test_load_parallel_module_from_multiple_dirs(tmp_path: Path, minimal_model: MinimalParallelModule):
    (tmp_path / "layer_3").mkdir()
    minimal_model.save_checkpoint(tmp_path, separate_file_for_parameters=None)

    # check the original loading mechanism
    minimal_model.load_checkpoint(
        tmp_path,
        allowed_missing_keys_in_checkpoint=[],
        allowed_unexpected_keys_in_checkpoint=[],
        ignore_keys_in_checkpoint=None,
    )

    # move the layer 3 to a subdirectory to test the loading functionality
    source_file = tmp_path / "model_state_layer_3_MinimalLayerNorm.pt"
    destination_dir = tmp_path / "layer_3"
    destination_file = destination_dir / source_file.name
    source_file.rename(destination_file)

    # reload checkpoint (with options) and save another one
    minimal_model.load_checkpoint(
        [tmp_path, destination_dir],
        allowed_missing_keys_in_checkpoint=[],
        allowed_unexpected_keys_in_checkpoint=[],
        ignore_keys_in_checkpoint=None,
    )


def test_load_parallel_module_from_multiple_dirs_where_one_is_missing(
    tmp_path: Path, minimal_model: MinimalParallelModule
):
    minimal_model.save_checkpoint(tmp_path, separate_file_for_parameters=None)

    wrong_path = tmp_path / "wrong_path"

    with pytest.raises(RuntimeError):
        minimal_model.load_checkpoint(
            [tmp_path, wrong_path],
            allowed_missing_keys_in_checkpoint=[],
            allowed_unexpected_keys_in_checkpoint=[],
            ignore_keys_in_checkpoint=None,
        )
