import torch

from scaling.core import ParallelModule, Topology
from scaling.core.logging import logger
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.utils import dist_launcher

from ..minimal import (
    MinimalConfig,
    MinimalContext,
    init_model,
)


def init_minimal_example_model(overwrite_config: dict) -> ParallelModule:
    """
    Helper function instantiating a parallel module class with the passed parallelization settings
    """
    # get configuration from launcher
    launch_config = LaunchConfig.from_launcher_args()
    config_dict = overwrite_config
    config_dict["topology"]["world_size"] = launch_config.world_size
    config_dict["topology"]["global_rank"] = launch_config.global_rank
    config_dict["topology"]["local_slot"] = launch_config.local_slot

    # initialize
    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = MinimalContext(config=config, topology=topology)
    logger.configure(
        config=config.logger,
        name=f"RANK {topology.config.global_rank}",
        global_rank=topology.config.global_rank,
    )
    logger.log_config(config=config)
    context.initialize(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        seed=config.trainer.seed,
    )

    # initialize model
    model = init_model(context=context)

    return model


def count_parameters(
    return_dict: dict,
    model_parallel_size: int,
    pipe_parallel_size: int,
):
    launch_config = LaunchConfig.from_launcher_args()
    config = MinimalConfig.from_dict(
        {
            "topology": {
                "world_size": launch_config.world_size,
                "model_parallel_size": model_parallel_size,
                "pipe_parallel_size": pipe_parallel_size,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
            },
        }
    )

    model = init_minimal_example_model(overwrite_config=config.as_dict())
    total_params, unique_params = model.get_params_count()
    return_dict["total_params"] = total_params
    return_dict["unique_params"] = unique_params


def test_parameter_count():
    """
    Tests if different parallelization layouts of model and pipe parallel size have the same number of unique params
    """
    param_counts = dict()

    # training param counts
    for world_size, model_parallel_size, pipe_parallel_size in [
        [1, 1, 1],
        [2, 1, 1],
        [2, 2, 1],
        [2, 1, 2],
        [4, 2, 2],
    ]:
        if world_size > torch.cuda.device_count():
            # cannot run test with world size greater than available cuda devices
            continue

        param_counts[f"ws_{world_size}_mp_{model_parallel_size}_pp_{pipe_parallel_size}"] = dist_launcher(
            run_func=count_parameters,
            world_size=world_size,
            master_port=find_free_port(),
            model_parallel_size=model_parallel_size,
            pipe_parallel_size=pipe_parallel_size,
        )

    # Get param count of first layout
    layout_ground_truth, params_ground_truth = next(iter(param_counts.items()))
    unique_params_key = "unique_params"
    unique_params_count_ground_truth = params_ground_truth[unique_params_key]

    # assertions
    for layout, params in param_counts.items():
        # make sure unique params are correct
        assert unique_params_count_ground_truth == params[unique_params_key], (
            f"{layout_ground_truth}: torch params {unique_params_count_ground_truth} differs from {layout}: "
            f"unique params count {params[unique_params_key]}"
        )
