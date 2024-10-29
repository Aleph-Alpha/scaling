from examples.mlp_example.config import MLPConfig
from examples.mlp_example.context import MLPContext
from examples.mlp_example.data import MNISTDataset
from examples.mlp_example.model import init_model, init_optimizer, loss_function, metrics_aggregation_fn
from scaling.core import BaseTrainer
from scaling.core.logging import logger
from scaling.core.runner import LaunchConfig
from scaling.core.topology import Topology


def main(launch_config: LaunchConfig) -> None:
    config_payload = launch_config.payload
    assert config_payload is not None
    topology_ = config_payload["topology"]
    assert topology_ is not None
    topology_["world_size"] = launch_config.world_size
    topology_["global_rank"] = launch_config.global_rank
    topology_["local_slot"] = launch_config.local_slot
    config = MLPConfig.from_dict(config_payload)

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
        loss_function=loss_function,  # type: ignore[arg-type]
    )

    trainer.run_training()


if __name__ == "__main__":
    launch_config = LaunchConfig.from_launcher_args()
    main(launch_config)
