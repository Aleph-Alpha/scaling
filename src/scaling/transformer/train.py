import os
import shutil
from pathlib import Path
from typing import Any, Callable

import torch

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    DeterminedBaseTrainer,
    Topology,
)
from scaling.core.logging import logger
from scaling.core.nn.parallel_module import EvaluationStepOutput, TrainStepOutput
from scaling.core.runner.launch_config import LaunchConfig
from scaling.transformer import (
    FinetuningChatBlendedDataset,
    FinetuningChatDataset,
    FinetuningTextBlendedDataset,
    FinetuningTextDataset,
    LegacyBlendedDataset,
    TextBlendedDataset,
    TextDataset,
    TransformerConfig,
    TransformerContext,
    TransformerParallelModule,
    init_model,
    init_optimizer,
)
from scaling.transformer.context.config import DataConfig
from scaling.transformer.dataset_loader import load_datasets
from scaling.transformer.model.model import loss_function, metrics_aggregation_fn
from scaling.transformer.utils.get_tflops import (
    get_model_flop_utilization_palm,
    get_tflops_aleph_alpha,
    get_tflops_bloom,
    get_tflops_electra,
    get_tflops_megatron,
)

try:
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import (
        ProfilerAgent as DeterminedProfilerAgent,  # type: ignore
    )
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = type(None)  # type: ignore
    DeterminedProfilerAgent = type(None)  # type: ignore


class TransformerTrainer(DeterminedBaseTrainer[TransformerContext, TransformerParallelModule]):
    def save_checkpoint(self, save_dir: Path | None = None) -> Path:
        save_dir = super().save_checkpoint(save_dir=save_dir)
        vocab_file = self.context.config.transformer_architecture.vocab_file
        if vocab_file is not None and save_dir is not None:
            shutil.copy(vocab_file, save_dir / "vocab.json")

        return save_dir

    def load_checkpoint(
        self,
        load_dir: Path | None = None,
        load_optimizer_states: bool = True,
        load_context: bool = True,
        allowed_missing_keys_in_checkpoint: list[str] | None = None,
        allowed_unexpected_keys_in_checkpoint: list[str] | None = None,
        ignore_keys_in_checkpoint: list[str] | None = None,
    ) -> bool:
        return super().load_checkpoint(
            load_dir=load_dir,
            load_optimizer_states=load_optimizer_states,
            load_context=load_context,
            allowed_missing_keys_in_checkpoint=allowed_missing_keys_in_checkpoint,
            allowed_unexpected_keys_in_checkpoint=allowed_unexpected_keys_in_checkpoint,
            ignore_keys_in_checkpoint=ignore_keys_in_checkpoint,
        )

    def log_metrics(
        self,
        train_step_output: TrainStepOutput,
        eval_step_output: EvaluationStepOutput | None = None,
    ) -> dict[str, Any]:
        logger.info(f"completed step {self.context.iterations}")
        metrics = self._train_step_output_log_metrics(train_step_output)
        metrics |= self._tflops_metrics(
            parameter_count=self.parameters_total,
            parameter_count_unique=self.parameters_unique,
            iter_time_s=train_step_output.step_duration,
        )
        if eval_step_output is not None:
            metrics |= self._eval_step_log_metrics(eval_step_output)
        logger.log_metrics(metrics, step=self.context.iterations)
        return metrics

    def _tflops_metrics(
        self, parameter_count: int, parameter_count_unique: int, iter_time_s: float
    ) -> dict[str, float]:
        architecture = self.context.config.transformer_architecture
        topology = self.context.topology
        tflops_dict = {}
        tflops_dict["runtime/tflops_megatron"] = get_tflops_megatron(
            parameter_count=parameter_count,
            iter_time_s=iter_time_s,
            topology=topology,
            transformer_architecture=architecture,
        )
        tflops_dict["runtime/tflops_megatron_layout_independent"] = get_tflops_megatron(
            parameter_count=parameter_count_unique,
            iter_time_s=iter_time_s,
            topology=topology,
            transformer_architecture=architecture,
        )
        tflops_dict["runtime/tflops_bloom"] = get_tflops_bloom(
            iter_time_s=iter_time_s,
            topology=topology,
            transformer_architecture=architecture,
        )
        tflops_dict["runtime/tflops_electra"] = get_tflops_electra(
            iter_time_s=iter_time_s,
            topology=topology,
            transformer_architecture=architecture,
        )
        tflops_dict["runtime/tflops_aleph_alpha"] = get_tflops_aleph_alpha(
            iter_time_s=iter_time_s,
            topology=topology,
            transformer_architecture=architecture,
        )
        tflops_dict["runtime/mfu_palm"] = get_model_flop_utilization_palm(
            iter_time_s=iter_time_s,
            parameter_count=parameter_count,
            topology=topology,
            transformer_architecture=architecture,
        )
        return tflops_dict

    def _train_step_output_log_metrics(self, tso: TrainStepOutput) -> dict[str, Any]:
        metrics = {
            "training/loss": tso.loss,
            "runtime/step_duration": tso.step_duration,
        }
        metrics |= tso.debug_dict if tso.debug_dict else {}
        metrics |= tso.metrics if tso.metrics else {}

        if tso.global_grad_norm is not None:
            metrics["training/global_grad_norm"] = tso.global_grad_norm
        if tso.global_grad_norm_clipped is not None:
            metrics["training/global_grad_norm_clipped"] = tso.global_grad_norm_clipped
        if tso.overflow is not None:
            metrics["training/overflow"] = int(tso.overflow)
        if tso.no_overflow_steps is not None:
            metrics["training/no_overflow_steps"] = tso.no_overflow_steps
        if tso.current_loss_scale is not None:
            metrics["training/current_loss_scale"] = tso.current_loss_scale

        learning_rates = tso.learning_rates if tso.learning_rates else {}
        for param_group_name, learning_rate in learning_rates.items():
            metrics[f"training/learning_rate_{param_group_name}"] = learning_rate

        return metrics

    def _eval_step_log_metrics(self, eval_step_output: EvaluationStepOutput) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        eval_step_output_metrics = eval_step_output.metrics if eval_step_output.metrics else {}
        for k, v in eval_step_output_metrics.items():
            metrics[f"evaluation/{k}"] = v
        metrics["evaluation/loss"] = eval_step_output.loss
        metrics["evaluation/step_duration"] = eval_step_output.step_duration
        return metrics


def main(
    launch_config: LaunchConfig,
    overwrite_config: dict[str, Any] | None = None,
    return_metrics: bool = False,
    determined_context: DeterminedContext | None = None,
    determined_profiler: DeterminedProfilerAgent | None = None,
) -> list[dict[str, Any]] | None:
    """
    Main function of the class. Runs training.
    Optionally returns list of losses.
    """
    config = _init_transformer_config(launch_config, overwrite_config)
    topology = Topology(config=config.topology)

    _init_logger(config, determined_context, topology)
    if config.training.use_deterministic_torch_algorithms:
        _enable_deterministic_torch()
    context = _init_transformer_context(config, determined_context, determined_profiler, launch_config, topology)
    model = init_model(context=context)
    optimizer = init_optimizer(context=context, model=model)

    blended_dataset: BaseDataset[Any, Any, Any] | None = None
    validation_blended_dataset: BaseDataset[Any, Any, Any] | None = None
    if topology.is_io_rank:
        blended_dataset, validation_blended_dataset = _read_datasets(context.config)

    trainer = TransformerTrainer(
        config=context.config.trainer,
        context=context,
        parallel_module=model,
        optimizer=optimizer,
        dataset=blended_dataset,
        sync_batch_to_model_parallel=_get_sync_batch(context.config.data),
        loss_function=loss_function,
        metrics_aggregation_fn=metrics_aggregation_fn,
        dataset_evaluation=validation_blended_dataset,
    )
    return trainer.run_training(return_metrics=return_metrics)


def _get_sync_batch(data_config: DataConfig) -> Callable:
    if data_config.finetuning_dataset:
        return FinetuningTextDataset.sync_batch_to_model_parallel
    if data_config.finetuning_chat_dataset:
        return FinetuningChatDataset.sync_batch_to_model_parallel
    return TextDataset.sync_batch_to_model_parallel


def _read_datasets(
    config: TransformerConfig,
) -> tuple[BaseDataset[Any, Any, Any] | None, BaseDataset[Any, Any, Any] | None]:
    validation_blended_dataset: BaseDataset[Any, Any, Any] | None = None
    architecture_config = config.transformer_architecture
    data_config = config.data
    logger.info("loading dataset")
    datasets, validation_datasets = load_datasets(data_config, architecture_config, config)

    seed = config.trainer.seed
    dataset_type = _get_dataset_type(data_config)
    blended_dataset = dataset_type(seed=seed, config=data_config.blended_dataset, datasets=datasets)
    if data_config.validation_data_prefixes:
        validation_blended_dataset = dataset_type(
            seed=seed, config=data_config.blended_dataset, datasets=validation_datasets
        )
    return blended_dataset, validation_blended_dataset


def _get_dataset_type(data_config: DataConfig) -> type[BaseBlendedDataset]:
    if data_config.legacy_dataset:
        return LegacyBlendedDataset
    if data_config.finetuning_dataset:
        return FinetuningTextBlendedDataset
    if data_config.finetuning_chat_dataset:
        return FinetuningChatBlendedDataset
    return TextBlendedDataset


def _init_transformer_context(
    config: TransformerConfig,
    determined_context: "DeterminedContext | None",
    determined_profiler: "DeterminedProfilerAgent | None",
    launch_config: LaunchConfig,
    topology: Topology,
) -> TransformerContext:
    context = TransformerContext(config=config, topology=topology)
    if determined_context is not None:
        context.initialize_with_determined(
            master_addr=launch_config.master_addr,
            master_port=str(launch_config.master_port),
            determined_context=determined_context,
            determined_profiler=determined_profiler,
            seed=config.trainer.seed,
        )
    else:
        context.initialize(
            master_addr=launch_config.master_addr,
            master_port=str(launch_config.master_port),
            seed=config.trainer.seed,
        )
    return context


def _init_logger(config: TransformerConfig, determined_context: DeterminedContext | None, topology: Topology) -> None:
    if config.runner.use_determined:
        logger.configure_determined(
            config=config.logger,
            name=f"RANK {topology.config.global_rank}",
            global_rank=topology.config.global_rank,
            determined_context=determined_context,
        )
    else:
        logger.configure(
            config=config.logger, name=f"RANK {topology.config.global_rank}", global_rank=topology.config.global_rank
        )
    logger.log_config(config=config)


def _init_transformer_config(launch_config: LaunchConfig, overwrite_config: dict[str, Any] | None) -> TransformerConfig:
    # overwrite config dict with launcher arguments
    config_dict = launch_config.payload or overwrite_config or {}
    config_dict = launch_config.overwrite_config_dict_with_launcher_args(config_dict)
    config = TransformerConfig.from_dict(config_dict, overwrite_values=overwrite_config)
    return config


def _enable_deterministic_torch() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # This is needed to enable deterministic CUBLA
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    main(LaunchConfig.from_launcher_args())
