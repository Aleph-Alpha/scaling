import argparse
import os
from typing import Union

from determined import ClusterInfo
from determined.core._context import Context as DeterminedContext  # type: ignore
from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore

from scaling.core.determined.core import init
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.determined_utils import determined_profiler_from_ctx, maybe_periodic_stacktraces

try:
    import determined as det  # type: ignore
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import (
        ProfilerAgent as DeterminedProfilerAgent,  # type: ignore
    )
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = None  # type: ignore
    DeterminedProfilerAgent = None  # type: ignore

from scaling.transformer import TransformerConfig
from scaling.transformer.train import main as train_main


def from_launcher_args_determined() -> LaunchConfig:
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    world_size = os.environ["WORLD_SIZE"]
    global_rank = os.environ["RANK"]
    local_slot = os.environ["LOCAL_RANK"]  # Torch distributed launcher set name as LOCAL_RANK

    parser = argparse.ArgumentParser(description="process launch")

    # Optional arguments for the launch helper
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config file",
    )
    parser.add_argument("remaining_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config is None:
        payload = None
    else:
        config = TransformerConfig.from_yaml(
            args.config, overwrite_values={"topology": {"world_size": int(world_size)}}
        )
        payload = config.as_dict()

    return LaunchConfig(
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        global_rank=global_rank,
        local_slot=local_slot,
        payload=payload,
    )


def main(
    determined_context: DeterminedContext,
    profiler: DeterminedProfilerAgent | None,
    overwrite_config: dict | None = None,
    return_metrics: bool = False,
    det_experiment_id: int | None = None,
    det_trial_id: int | None = None,
    info: ClusterInfo | None = None,
) -> list[dict[str, Union[float, int]]] | None:
    """
    Collects determined launcher arguments and calls training script
    """
    launch_config = from_launcher_args_determined()
    if overwrite_config is None:
        overwrite_config = dict()
    if "runner" not in overwrite_config:
        overwrite_config["runner"] = dict()
    overwrite_config["runner"]["use_determined"] = True
    overwrite_config["determined_experiment_id"] = det_experiment_id
    overwrite_config["determined_trial_id"] = det_trial_id

    print(f"Prior {overwrite_config=}")
    hparams = info.trial.hparams if info is not None else None

    if hparams and "layout" in hparams:
        # Layout is used for restarting individual runs and contains all the specific layouting settings
        # Overwrite hparams with this layout information
        hparams = hparams["layout"]

    if hparams is not None and len(hparams) > 0:
        print(f"Hparams before: {hparams}")

        for k in (
            "topology",
            "transformer_architecture",
            "logger",
            "trainer",
            "learning_rate_scheduler",
        ):
            if k not in overwrite_config:
                overwrite_config[k] = {}

        if hparams.get("learning_rate"):
            overwrite_config["learning_rate_scheduler"]["learning_rate"] = hparams["learning_rate"]

        if hparams.get("target_train_tokens"):
            assert "sequence_length" in hparams
            assert "global_batch_size" in hparams
            train_iterations = int(
                hparams["target_train_tokens"] / (hparams["global_batch_size"] * hparams["sequence_length"])
            )
            overwrite_config["trainer"]["train_iterations"] = train_iterations

        if hparams.get("warmup_tokens"):
            assert "sequence_length" in hparams
            assert "global_batch_size" in hparams
            warmup_steps = int(hparams["warmup_tokens"] / (hparams["global_batch_size"] * hparams["sequence_length"]))
            overwrite_config["learning_rate_scheduler"]["learning_rate_warmup_steps"] = warmup_steps

        if "wandb_project" in hparams:
            overwrite_config["logger"]["wandb_project"] = hparams["wandb_project"]

        if "model_parallel_size" in hparams:
            overwrite_config["topology"]["model_parallel_size"] = hparams["model_parallel_size"]
        if "pipe_parallel_size" in hparams:
            overwrite_config["topology"]["pipe_parallel_size"] = hparams["pipe_parallel_size"]
        if "sequence_parallel" in hparams:
            overwrite_config["topology"]["sequence_parallel"] = hparams["sequence_parallel"]

        if "global_batch_size" in hparams:
            overwrite_config["topology"]["global_batch_size"] = hparams["global_batch_size"]

        if "train_iterations" in hparams:
            overwrite_config["trainer"]["train_iterations"] = hparams["train_iterations"]
        if "micro_batch_size" in hparams:
            overwrite_config["topology"]["micro_batch_size"] = hparams["micro_batch_size"]
        if "activation_checkpointing_type" in hparams:
            overwrite_config["topology"]["activation_checkpointing_type"] = hparams["activation_checkpointing_type"]
        if "pipe_partition_method" in hparams:
            overwrite_config["topology"]["pipe_partition_method"] = hparams["pipe_partition_method"]
        if "pipe_partition_overwrite" in hparams:
            overwrite_config["topology"]["pipe_partition_overwrite"] = hparams["pipe_partition_overwrite"]
        if "kernel" in hparams:
            overwrite_config["transformer_architecture"]["masked_softmax"] = {}
            overwrite_config["transformer_architecture"]["masked_softmax"]["kernel"] = hparams["kernel"]

        if "sequence_length" in hparams:
            overwrite_config["transformer_architecture"]["sequence_length"] = hparams["sequence_length"]

        print(f"{overwrite_config=}")

    return train_main(
        launch_config,
        overwrite_config,
        return_metrics,
        determined_context,
        profiler,
    )


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info is not None
    config_determined = det.ExperimentConfig(info.trial._config)
    det_experiment_id = info.trial.experiment_id
    det_trial_id = info.trial.trial_id

    distributed = det.core.DistributedContext.from_torch_distributed()
    with maybe_periodic_stacktraces(config_determined.debug_enabled()):
        with init(distributed=distributed) as determined_context:  # type: ignore[arg-type]
            with determined_profiler_from_ctx(determined_context, config_determined, info) as profiler:
                main(
                    determined_context,
                    profiler,
                    det_experiment_id=det_experiment_id,
                    det_trial_id=det_trial_id,
                    info=info,
                )
