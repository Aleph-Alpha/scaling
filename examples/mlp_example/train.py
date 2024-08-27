from scaling.core.logging import logger
from scaling.core.runner import LaunchConfig
from scaling.core.topology import Topology
from scaling.core import BaseTrainer

from examples.mlp_example.config import MLPConfig
from examples.mlp_example.context import MLPContext
from examples.mlp_example.data import MNISTDataset
from examples.mlp_example.model import init_model
from examples.mlp_example.model import init_optimizer
from examples.mlp_example.model import loss_function
from examples.mlp_example.model import metrics_aggregation_fn


def main(launch_config: LaunchConfig):
    config = launch_config.payload
    config["topology"]["world_size"] = launch_config.world_size
    config["topology"]["global_rank"] = launch_config.global_rank
    config["topology"]["local_slot"] = launch_config.local_slot
    config = MLPConfig.from_dict(launch_config.payload)

    topology = Topology(config=config.topology)
    context = MLPContext(config=config, topology=topology)

    logger.configure(
        config=config.logger,
        name=f"RANK {topology.config.global_rank}",
        global_rank=topology.config.global_rank,
    )

    context.initialize(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        seed=config.trainer.seed,
    )

    model = init_model(context=context)
    optimizer = init_optimizer(context=context, model=model)

    train_data = None
    valid_data = None
    if topology.is_io_rank:
        train_data = MNISTDataset(train=True)
        valid_data = MNISTDataset(train=False)

    trainer = BaseTrainer(
        config=context.config.trainer,
        context=context,
        parallel_module=model,
        optimizer=optimizer,
        dataset=train_data,
        dataset_evaluation=valid_data,
        sync_batch_to_model_parallel=MNISTDataset.sync_batch_to_model_parallel,
        metrics_aggregation_fn=metrics_aggregation_fn,
        loss_function=loss_function,
    )

    trainer.run_training()


if __name__ == "__main__":
    launch_config = LaunchConfig.from_launcher_args()
    main(launch_config)
