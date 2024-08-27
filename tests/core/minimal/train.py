from pathlib import Path
from typing import Dict, List, Optional, Union

from scaling.core import BaseTrainer, Topology
from scaling.core.logging import logger
from scaling.core.runner.launch_config import LaunchConfig

from . import (
    MinimalConfig,
    MinimalContext,
    MinimalDataset,
    init_model,
    init_optimizer,
    loss_function,
    metrics_aggregation_fn,
)
from .model.model import MinimalParallelModule


class MinimalTrainer(BaseTrainer[MinimalContext, MinimalParallelModule]):
    def load_checkpoint(
        self,
        load_dir: Optional[Path] = None,
        load_optimizer_states: bool = True,
        load_context: bool = True,
        allowed_missing_keys_in_checkpoint: Optional[List[str]] = None,
        allowed_unexpected_keys_in_checkpoint: Optional[List[str]] = None,
        ignore_keys_in_checkpoint: Optional[List[str]] = None,
    ) -> bool:
        return super().load_checkpoint(
            load_dir=load_dir,
            load_optimizer_states=load_optimizer_states,
            load_context=load_context,
            allowed_missing_keys_in_checkpoint=allowed_missing_keys_in_checkpoint,
            allowed_unexpected_keys_in_checkpoint=allowed_unexpected_keys_in_checkpoint,
            ignore_keys_in_checkpoint=ignore_keys_in_checkpoint,
        )


def main(
    overwrite_config: Optional[dict] = None, return_metrics: bool = False
) -> Optional[List[Dict[str, Union[float, int]]]]:
    """
    Main function of the class. Runs training.
    Optionally returns list of losses.
    """
    # get configuration from launcher
    launch_config = LaunchConfig.from_launcher_args()
    config_dict = launch_config.payload or overwrite_config or dict()
    config_dict["topology"]["world_size"] = launch_config.world_size
    config_dict["topology"]["global_rank"] = launch_config.global_rank
    config_dict["topology"]["local_slot"] = launch_config.local_slot
    if (
        config_dict.get("profiler", dict()).get("profiler_output") is None
        and config_dict.get("logger", dict()).get("log_dir") is not None
    ):
        config_dict["profiler"]["profiler_output"] = Path(config_dict["logger"]["log_dir"]) / "profile.json"

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

    # initialize model, optimizer and data loader
    model = init_model(context=context)
    optimizer = init_optimizer(context=context, model=model)
    model.get_params_count()
    if topology.is_io_rank:
        dataset: Optional[MinimalDataset] = MinimalDataset(seed=context.config.trainer.seed)
    else:
        dataset = None

    trainer = MinimalTrainer(
        config=context.config.trainer,
        context=context,
        parallel_module=model,
        optimizer=optimizer,
        dataset=dataset,
        sync_batch_to_model_parallel=MinimalDataset.sync_batch_to_model_parallel,
        loss_function=loss_function,
        metrics_aggregation_fn=metrics_aggregation_fn,
    )
    losses = trainer.run_training(return_metrics=return_metrics)
    return losses


if __name__ == "__main__":
    main()
