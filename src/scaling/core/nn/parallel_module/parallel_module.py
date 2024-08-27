from typing import (
    Any,
    Callable,
    Generic,
    NamedTuple,
    Optional,
    Union,
)

import torch
from torch.utils._pytree import tree_map

from scaling.core.topology.topology_config import ActivationCheckpointingType

from ...data import BaseLayerIO, DataLoader
from ...optimizer import BaseOptimizer
from ...optimizer.allreduce import allreduce_tensor_in_float32
from ...profiler import Profiler, ProfilerConfig, SynchronizedTimer
from ...topology import Topology
from ..parameter_meta import CoreParameterMeta
from ..pipeline_schedule import (
    PipelineScheduleInference,
    PipelineScheduleTrain,
)
from ..pipeline_schedule.instructions import (
    InstructionBackwardPass,
    InstructionBase,
    InstructionForwardPass,
    InstructionLoadMicroBatch,
    InstructionLoss,
    InstructionOptimizerStep,
    InstructionRecvActivation,
    InstructionRecvGrad,
    InstructionReduceTiedGrads,
    InstructionSendActivation,
    InstructionSendGrad,
    InstructionStoreMicroBatch,
)
from .activation_checkpointing import _checkpoint_without_reentrant
from .base_layer import (
    BaseDatasetBatchGeneric,
    BaseLossInputGeneric,
)
from .buffers import Buffers, BufferType
from .communicator import PipeCommunicator
from .layer_spec import LayerSpec
from .partitioned_module import PipePartitionedModule
from .tied_layer_index import TiedLayerIndex


def get_timer_args(
    instruction: InstructionBase,
) -> tuple[str, Optional[int], Optional[int]]:
    """Given an instruction, returns appropriate arguments for `Profiler.time()`."""
    timer_name = instruction.__class__.__name__.lstrip("Instruction")
    micro_batch_id = instruction.micro_batch_id
    buffer_id = instruction.buffer_id
    if instruction.__class__ in [
        InstructionReduceTiedGrads,
        InstructionOptimizerStep,
    ]:
        micro_batch_id = buffer_id = -1
    return timer_name, micro_batch_id, buffer_id


class TrainStepOutput(NamedTuple):
    loss: Optional[float]
    metrics: Optional[dict[str, Union[int, float]]]
    global_grad_norm: Optional[float]
    global_grad_norm_clipped: Optional[float]
    learning_rates: Optional[dict[str, float]]
    overflow: Optional[bool]
    no_overflow_steps: Optional[int]
    current_loss_scale: Optional[float]
    step_duration: float
    debug_dict: Optional[dict[str, float]]


class EvaluationStepOutput(NamedTuple):
    loss: Optional[float]
    metrics: Optional[dict[str, int | float]]
    step_duration: float


def map_value_from_tensor_fn(x: Any) -> Any:
    return x.tolist() if torch.is_tensor(x) else x


class ParallelModule(
    PipePartitionedModule,
    Generic[BaseLossInputGeneric, BaseDatasetBatchGeneric],
):
    def __init__(
        self,
        layer_specs: list[LayerSpec],
        topology: Topology,
        profiler_config: ProfilerConfig = ProfilerConfig(),
        use_continuous_recommunication: bool = False,
    ):
        self.topology: Topology

        super().__init__(
            layer_specs=layer_specs,
            topology=topology,
        )
        self._layer_specs = layer_specs
        self.tied_layer_index: TiedLayerIndex

        # Precompute train instructions (won't change unless topology changes).
        self.train_schedule = PipelineScheduleTrain(
            topology=self.topology,
        )
        self.train_instructions: list[InstructionBase] = self.train_schedule.instructions()
        self.train_required_buffer_count = self.train_schedule.required_buffer_count()

        self.evaluation_schedule = PipelineScheduleInference(topology=self.topology)
        self.evaluation_instructions: list[InstructionBase] = self.evaluation_schedule.instructions()
        self.evaluation_required_buffer_count = self.evaluation_schedule.required_buffer_count()

        # Initialize class holding pipeline buffers.
        self.pipe_buffer = Buffers()

        # Initialize communicators for inflow to pipe stage and outflow of pipe stage.
        self.communicator_in = PipeCommunicator(
            local_device=self.topology.device,
            recv_grads=False,
            recv_data=True,
            use_continuous_recommunication=use_continuous_recommunication,
        )
        self.communicator_out = PipeCommunicator(
            local_device=self.topology.device,
            recv_grads=True,
            recv_data=False,
            use_continuous_recommunication=use_continuous_recommunication,
        )
        self.communicator_loss_in: Optional[PipeCommunicator] = None
        self.communicator_loss_out: Optional[PipeCommunicator] = None
        if self.topology.config.pipe_parallel_size > 1:
            if self.topology.is_first_pipe_parallel_rank:
                self.communicator_loss_in = PipeCommunicator(
                    local_device=self.topology.device,
                    recv_grads=False,
                    recv_data=True,
                )
            if self.topology.is_last_pipe_parallel_rank:
                self.communicator_loss_out = PipeCommunicator(
                    local_device=self.topology.device,
                    recv_grads=False,
                    recv_data=False,
                )

        # Initialize profiler.
        self.profiler = Profiler(config=profiler_config, topology=self.topology)
        self.step_timer = SynchronizedTimer()

        # Sync model.
        self.broadcast_model()

    def named_parameters_with_meta(
        self,
    ) -> list[tuple[str, torch.Tensor, CoreParameterMeta]]:
        named_parameters_with_meta = list()

        for local_layer_index, layer in enumerate(self._layers):
            layer_index = self._pipe_partition_coordinates[0].start + local_layer_index
            local_tied_duplicates = self.tied_layer_index.layer_index_to_tied_local_duplicate_parameter_names(
                layer_index=layer_index
            )

            for parameter_name, parameter in layer.named_parameters():
                if parameter_name in local_tied_duplicates:
                    continue
                meta: CoreParameterMeta = parameter.core_parameter_meta
                named_parameters_with_meta.append((parameter_name, parameter, meta))
        return named_parameters_with_meta

    def broadcast_model(self) -> None:
        """Broadcasts model weights from data parallel rank 0 to all other data parallel ranks."""
        if self.topology is None or not self.topology.is_distributed_initialized:
            return

        for parameter in self._layers.parameters():
            if torch.is_tensor(parameter):
                # Broadcast duplicated parameters to model parallel.
                if parameter.core_parameter_meta.is_model_parallel_duplicate:  # type: ignore[attr-defined]
                    torch.distributed.broadcast(
                        parameter,
                        torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
                        group=self.topology.model_parallel_group,
                    )

                # Broadcast to data parallel.
                torch.distributed.broadcast(
                    parameter,
                    torch.distributed.distributed_c10d.get_global_rank(self.topology.data_parallel_group, 0),
                    group=self.topology.data_parallel_group,
                )

        # Broadcast to tied layers.
        for (
            parameter,
            process_group,
            pipe_parallel_ranks,
        ) in self.tied_layer_index.local_parameters_and_process_groups():
            if len(pipe_parallel_ranks) > 1:
                torch.distributed.broadcast(
                    parameter,
                    torch.distributed.distributed_c10d.get_global_rank(process_group, 0),  # type: ignore[arg-type]
                    group=process_group,
                )

    def get_params_count(self) -> tuple[int, int]:
        """Returns the number of total and unique parameters of a model.

        Requires parameters to have `core_parameter_meta` property which stores the information
        if a parameter is duplicated across model parallel ranks.
        """
        params = 0
        unique_params = 0
        if self.topology.data_parallel_rank == 0:
            for local_layer_index, layer in enumerate(self._layers):
                is_tied_duplicate = self.tied_layer_index.layer_index_is_tied_global_duplicate(
                    layer_index=self._pipe_partition_coordinates[0].start + local_layer_index
                )

                for parameter in layer.parameters():
                    p_count = parameter.nelement()
                    params += p_count

                    is_model_parallel_duplicate = self.topology.model_parallel_rank != 0 and (
                        parameter.core_parameter_meta.is_model_parallel_duplicate
                    )
                    if not (is_tied_duplicate or is_model_parallel_duplicate):
                        unique_params += p_count

        total_n_parameters = torch.tensor([params, unique_params]).cuda()
        torch.distributed.all_reduce(total_n_parameters)
        params = int(total_n_parameters[0].item())
        unique_params = int(total_n_parameters[1].item())
        return params, unique_params

    def _forward_tuple_input(self, *args: tuple) -> torch.nn.Module:
        x = self._layers[0].tuple_to_input(tuple(args))
        for layer in self._layers:
            x = layer(x)
        return x

    def forward(self, x: BaseLayerIO) -> BaseLayerIO:
        if self.training and (
            self.topology.config.activation_checkpointing_type == ActivationCheckpointingType.EVERY_PIPE_STAGE
        ):
            # Call block with torch activation checkpointing.
            x = _checkpoint_without_reentrant(
                self._forward_tuple_input,
                self.topology,
                True,
                *self._layers[0].input_to_tuple(x),
            )

        elif self.training and (
            self.topology.config.activation_checkpointing_type == ActivationCheckpointingType.EVERY_LAYER
        ):
            for layer in self._layers:
                # Call every layer with torch activation checkpointing.
                x = _checkpoint_without_reentrant(
                    layer._forward_tuple_input,
                    self.topology,
                    True,
                    *layer.input_to_tuple(x),
                )
        else:
            for layer in self._layers:
                x = layer(x)
        return x

    def get_loss(
        self,
        metrics_aggregation_fn: Optional[
            Callable[
                [Topology, Optional[list[dict[str, torch.Tensor]]]],
                dict[str, torch.Tensor],
            ]
        ],
    ) -> tuple[Optional[float], Optional[dict[str, float]]]:
        loss: torch.Tensor
        if self.topology.is_last_pipe_parallel_rank:
            assert self.pipe_buffer.accum_loss is not None
            loss = self.pipe_buffer.accum_loss / self.topology.config.gradient_accumulation_steps
            metrics_list = list(self.pipe_buffer.dump(BufferType.METRICS).values())

            if len(metrics_list) > 0:
                assert metrics_aggregation_fn is not None
                metrics: Optional[dict[str, torch.Tensor]] = metrics_aggregation_fn(self.topology, metrics_list)
            else:
                metrics = None

            assert self.topology.config.data_parallel_size is not None
            if self.topology.config.data_parallel_size > 1:
                torch.distributed.all_reduce(loss, group=self.topology.data_parallel_group)
                loss = loss / self.topology.config.data_parallel_size

            data = (loss, metrics)

        if self.topology.config.pipe_parallel_size > 1:
            if self.topology.is_first_pipe_parallel_rank:
                assert self.communicator_loss_in is not None

                _data = self.communicator_loss_in.recv_data(
                    origin_global_rank=self.topology.get_global_rank(pipe_parallel_rank=-1)
                )
                loss, metrics = _data

                return loss.cpu().item(), tree_map(map_value_from_tensor_fn, metrics)

            if self.topology.is_last_pipe_parallel_rank:
                assert self.communicator_loss_out is not None

                self.communicator_loss_out.send_data(
                    data=data,
                    target_global_rank=self.topology.get_global_rank(pipe_parallel_rank=0),
                )

                return None, None

            # still return a tuple if not first or last pipe rank
            return None, None

        else:
            return loss.cpu().item(), tree_map(map_value_from_tensor_fn, metrics)

    def train_step(
        self,
        dataloader: Optional[DataLoader],
        optimizer: BaseOptimizer,
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
        loss_function: Callable[
            [BaseLossInputGeneric, Optional[BaseDatasetBatchGeneric]],
            Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]],
        ],
        metrics_aggregation_fn: Optional[Callable],
    ) -> TrainStepOutput:
        if not torch._C.is_grad_enabled():
            raise RuntimeError("train_step() requires gradients enabled. Use evaluation_step() instead.")

        self.step_timer.reset()
        self.step_timer.start()
        self._layers.train()
        self.pipe_buffer.reset()
        self.profiler.step()

        # Execute the instructions.
        for instruction in self.train_instructions:
            io_buffer_id = instruction.buffer_id
            timer_name, micro_batch_id, buffer_id = get_timer_args(instruction)
            with self.profiler.time(timer_name, micro_batch_id, buffer_id):
                if instruction.__class__ == InstructionLoadMicroBatch:
                    assert io_buffer_id is not None
                    self._execute_load_micro_batch(
                        dataloader=dataloader,
                        io_buffer_id=io_buffer_id,
                        sync_batch_to_model_parallel=sync_batch_to_model_parallel,
                    )
                elif instruction.__class__ == InstructionForwardPass:
                    assert io_buffer_id is not None
                    assert instruction.buffer_id is not None
                    self._execute_forward_pass(
                        io_buffer_id=io_buffer_id,
                        buffer_id=instruction.buffer_id,
                    )
                elif instruction.__class__ == InstructionLoss:
                    assert io_buffer_id is not None
                    assert instruction.buffer_id is not None
                    assert instruction.is_first_pass is not None
                    self._execute_loss_fn(
                        buffer_id=instruction.buffer_id,
                        io_buffer_id=io_buffer_id,
                        is_first_pass=instruction.is_first_pass,
                        loss_function=loss_function,
                    )
                elif instruction.__class__ == InstructionBackwardPass:
                    assert instruction.buffer_id is not None
                    self._execute_backward_pass(buffer_id=instruction.buffer_id, optimizer=optimizer)
                elif instruction.__class__ == InstructionSendActivation:
                    assert instruction.buffer_id is not None
                    self._execute_send_activations(
                        buffer_id=instruction.buffer_id,
                    )
                elif instruction.__class__ == InstructionRecvActivation:
                    assert io_buffer_id is not None
                    self._execute_receive_activations(buffer_id=io_buffer_id)
                elif instruction.__class__ == InstructionSendGrad:
                    assert io_buffer_id is not None
                    self._execute_send_gradients(buffer_id=io_buffer_id)
                elif instruction.__class__ == InstructionRecvGrad:
                    assert instruction.buffer_id is not None
                    self._execute_receive_gradients(buffer_id=instruction.buffer_id)
                elif instruction.__class__ == InstructionReduceTiedGrads:
                    self._execute_reduce_tied_grads()
                elif instruction.__class__ == InstructionOptimizerStep:
                    optimizer_step_output = (
                        optimizer.step()
                    )  # Including optimizer step here to be able to interim accumulate grads later
                else:
                    raise NotImplementedError(f"Instruction '{instruction.__class__.__name__}' not implemented")

        loss, metrics = self.get_loss(metrics_aggregation_fn=metrics_aggregation_fn)
        self.profiler.flush()
        self.step_timer.stop()
        return TrainStepOutput(
            loss=loss,
            metrics=metrics,
            step_duration=self.step_timer.duration(),
            **optimizer_step_output._asdict(),
        )

    def evaluation_step(
        self,
        dataloader: Optional[DataLoader],
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
        loss_function: Callable[
            [BaseLossInputGeneric, Optional[BaseDatasetBatchGeneric]],
            Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]],
        ],
        metrics_aggregation_fn: Optional[Callable],
    ) -> EvaluationStepOutput:
        self.step_timer.reset()
        self.step_timer.start()
        self._layers.eval()
        self.pipe_buffer.reset()
        self.profiler.step()

        for instruction in self.evaluation_instructions:
            timer_name, micro_batch_id, buffer_id = get_timer_args(instruction)
            with self.profiler.time(timer_name, micro_batch_id, buffer_id):
                if instruction.__class__ == InstructionLoadMicroBatch:
                    assert instruction.buffer_id is not None
                    self._execute_load_micro_batch(
                        dataloader=dataloader,
                        io_buffer_id=instruction.buffer_id,
                        sync_batch_to_model_parallel=sync_batch_to_model_parallel,
                    )
                elif instruction.__class__ == InstructionForwardPass:
                    assert instruction.buffer_id is not None
                    with torch.no_grad():
                        self._execute_forward_pass(
                            io_buffer_id=instruction.buffer_id,
                            buffer_id=instruction.buffer_id,
                        )
                elif instruction.__class__ == InstructionLoss:
                    assert instruction.buffer_id is not None
                    with torch.no_grad():
                        self._execute_loss_fn(
                            buffer_id=instruction.buffer_id,
                            io_buffer_id=instruction.buffer_id,
                            is_first_pass=True,
                            loss_function=loss_function,
                        )
                elif instruction.__class__ == InstructionSendActivation:
                    assert instruction.buffer_id is not None
                    self._execute_send_activations(buffer_id=instruction.buffer_id)
                elif instruction.__class__ == InstructionRecvActivation:
                    assert instruction.buffer_id is not None
                    self._execute_receive_activations(buffer_id=instruction.buffer_id)
                else:
                    raise NotImplementedError(f"Instruction '{instruction.__class__.__name__}' not implemented")

        loss, metrics = self.get_loss(metrics_aggregation_fn=metrics_aggregation_fn)
        self.profiler.flush()
        self.step_timer.stop()
        return EvaluationStepOutput(
            loss=loss,
            metrics=metrics,
            step_duration=self.step_timer.duration(),
        )

    def run_instructions(
        self,
        instructions: list[InstructionBase],
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
        collect_outputs_from_model_parallel: Callable[[Topology, BaseLossInputGeneric], BaseLossInputGeneric],
        batch: Optional[BaseDatasetBatchGeneric] = None,
    ) -> BaseLossInputGeneric | None:
        for instruction in instructions:
            if instruction.__class__ == InstructionStoreMicroBatch:
                assert instruction.buffer_id is not None
                self._execute_store_micro_batch(
                    batch=batch,
                    buffer_id=instruction.buffer_id,
                    sync_batch_to_model_parallel=sync_batch_to_model_parallel,
                )
            elif instruction.__class__ == InstructionForwardPass:
                assert instruction.buffer_id is not None
                self._execute_forward_pass(
                    io_buffer_id=instruction.buffer_id,
                    buffer_id=instruction.buffer_id,
                    take_input=True,
                )
            elif instruction.__class__ == InstructionSendActivation:
                assert instruction.buffer_id is not None
                self._execute_send_activations(buffer_id=instruction.buffer_id)
            elif instruction.__class__ == InstructionRecvActivation:
                assert instruction.buffer_id is not None
                self._execute_receive_activations(buffer_id=instruction.buffer_id)
            else:
                raise NotImplementedError(
                    f"Instruction '{instruction.__class__.__name__}' not implemented for run_instructions."
                )

        if batch is not None and self.topology.config.pipe_parallel_size == 1:
            assert instruction.buffer_id is not None
            outputs = self.pipe_buffer.take(
                buffer_type=BufferType.PIPELINE_STAGE_OUTPUT,
                buffer_id=instruction.buffer_id,
            )
            outputs = collect_outputs_from_model_parallel(self.topology, outputs)
        elif (
            self.topology.config.pipe_parallel_size > 1
            and self.topology.pipe_parallel_rank == 0
            and len(instructions) > 0
            and isinstance(instructions[-1], InstructionRecvActivation)
        ):
            assert instruction.buffer_id is not None
            # we are getting the output of the last pipe rank into the input of the current rank
            outputs = self.pipe_buffer.take(
                buffer_type=BufferType.PIPELINE_STAGE_INPUT,
                buffer_id=instruction.buffer_id,
            )
            outputs = collect_outputs_from_model_parallel(self.topology, outputs)
        else:
            outputs = None

        return outputs

    def _execute_store_micro_batch(
        self,
        batch: Optional[BaseDatasetBatchGeneric],
        buffer_id: int,
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
    ) -> None:
        """Loads a micro batch and writes to the buffer."""
        batch = sync_batch_to_model_parallel(self.topology, batch)
        self.pipe_buffer.write(
            buffer_type=BufferType.PIPELINE_STAGE_INPUT,
            buffer_id=buffer_id,
            data=batch,
        )

    def _execute_load_micro_batch(
        self,
        dataloader: Optional[DataLoader],
        io_buffer_id: int,
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
    ) -> None:
        """Loads a micro batch and writes to the buffer."""
        if self.topology.is_io_rank:
            assert dataloader is not None
            batch = next(dataloader)
        else:
            batch = None

        batch = sync_batch_to_model_parallel(self.topology, batch)

        # Data is only needed on first and last pipe stage.
        if self.topology.is_first_pipe_parallel_rank:
            self.pipe_buffer.write(
                buffer_type=BufferType.PIPELINE_STAGE_INPUT,
                buffer_id=io_buffer_id,
                data=batch.only_inputs(),
            )
        if self.topology.is_last_pipe_parallel_rank:
            self.pipe_buffer.write(
                buffer_type=BufferType.TARGET,
                buffer_id=io_buffer_id,
                data=batch.only_targets(),
            )

    def _execute_forward_pass(
        self,
        io_buffer_id: int,
        buffer_id: int,
        take_input: bool = False,
    ) -> None:
        """Loads a micro batch and writes to the buffer."""
        if take_input:
            pipeline_stage_input = self.pipe_buffer.take(
                buffer_type=BufferType.PIPELINE_STAGE_INPUT, buffer_id=io_buffer_id
            )
        else:
            pipeline_stage_input = self.pipe_buffer.get(
                buffer_type=BufferType.PIPELINE_STAGE_INPUT, buffer_id=io_buffer_id
            )
        outputs = self(pipeline_stage_input)
        self.pipe_buffer.write(
            buffer_type=BufferType.PIPELINE_STAGE_OUTPUT,
            buffer_id=buffer_id,
            data=outputs,
        )

    def _execute_loss_fn(
        self,
        buffer_id: int,
        io_buffer_id: int,
        is_first_pass: bool,
        loss_function: Callable[
            [BaseLossInputGeneric, Optional[BaseDatasetBatchGeneric]],
            Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]],
        ],
    ) -> None:
        """Computes loss on last stage.

        If no loss function is given, we assume the final layer computes a loss.
        """
        outputs = self.pipe_buffer.get(buffer_type=BufferType.PIPELINE_STAGE_OUTPUT, buffer_id=buffer_id)

        if is_first_pass:
            target = self.pipe_buffer.get(buffer_type=BufferType.TARGET, buffer_id=io_buffer_id)
        else:
            target = self.pipe_buffer.take(buffer_type=BufferType.TARGET, buffer_id=io_buffer_id)

        loss_output = loss_function(outputs, target)

        if isinstance(loss_output, tuple):
            loss: torch.Tensor = loss_output[0]
            metrics: dict[str, torch.Tensor] = loss_output[1]

            self.pipe_buffer.write(
                buffer_type=BufferType.METRICS,
                buffer_id=buffer_id,
                data=metrics,
            )

        else:
            loss = loss_output

        assert torch.is_tensor(loss)
        assert (
            list(loss.size()) == []
        ), f"The loss needs to be a scalar. The loss currently has the shape: {loss.size()}"

        self.pipe_buffer.add_loss(loss)
        self.pipe_buffer.write(
            buffer_type=BufferType.LOSS,
            buffer_id=buffer_id,
            data=loss,
        )

    def _execute_backward_pass(self, buffer_id: int, optimizer: BaseOptimizer) -> None:
        if self.topology.is_last_pipe_parallel_rank:
            loss: torch.Tensor = self.pipe_buffer.take(buffer_type=BufferType.LOSS, buffer_id=buffer_id)
            optimizer.backward(loss)
        else:
            grad = self.pipe_buffer.take(buffer_type=BufferType.GRAD, buffer_id=buffer_id)
            torch.autograd.backward(
                tensors=grad.tensors,
                grad_tensors=grad.grad_tensors,
            )

    def _execute_send_activations(
        self,
        buffer_id: int,
    ) -> None:
        outputs = self.pipe_buffer.get(buffer_type=BufferType.PIPELINE_STAGE_OUTPUT, buffer_id=buffer_id)
        outputs_tuple = self._layers[-1].output_to_tuple(outputs)

        # Make next rank the first in case of inference and last pipeline rank.
        next_pipe_parallel_rank = self.topology.next_pipe_parallel_rank
        if next_pipe_parallel_rank is None:
            next_pipe_parallel_rank = 0

        self.communicator_out.send_data(
            outputs_tuple,
            target_global_rank=self.topology.get_global_rank(pipe_parallel_rank=next_pipe_parallel_rank),
        )

    def _execute_receive_activations(self, buffer_id: int) -> None:
        # Make next rank the last in case of inference and first pipeline rank.
        previous_pipe_parallel_rank = self.topology.previous_pipe_parallel_rank
        if previous_pipe_parallel_rank is None:
            previous_pipe_parallel_rank = self.topology.config.pipe_parallel_size - 1
        inputs_tuple = self.communicator_in.recv_data(
            origin_global_rank=self.topology.get_global_rank(pipe_parallel_rank=previous_pipe_parallel_rank)
        )

        if self.topology.is_first_pipe_parallel_rank:
            inputs = self._layers[0].tuple_to_last_stage_activation(inputs_tuple)
        else:
            inputs = self._layers[0].tuple_to_input(inputs_tuple)
        self.pipe_buffer.write(
            buffer_type=BufferType.PIPELINE_STAGE_INPUT,
            buffer_id=buffer_id,
            data=inputs,
        )

    def _execute_send_gradients(self, buffer_id: int) -> None:
        inputs = self.pipe_buffer.take(buffer_type=BufferType.PIPELINE_STAGE_INPUT, buffer_id=buffer_id)
        inputs_tuple = self._layers[0].input_to_tuple(inputs)
        self.communicator_in.send_gradients(
            inputs_tuple,
            target_global_rank=self.topology.get_global_rank(
                pipe_parallel_rank=self.topology.previous_pipe_parallel_rank
            ),
        )

    def _execute_receive_gradients(self, buffer_id: int) -> None:
        outputs = self.pipe_buffer.take(buffer_type=BufferType.PIPELINE_STAGE_OUTPUT, buffer_id=buffer_id)
        outputs_tuple = self._layers[-1].output_to_tuple(outputs)
        grad = self.communicator_out.recv_gradients(
            outputs_tuple,
            origin_global_rank=self.topology.get_global_rank(pipe_parallel_rank=self.topology.next_pipe_parallel_rank),
        )
        self.pipe_buffer.write(buffer_type=BufferType.GRAD, buffer_id=buffer_id, data=grad)

    def _execute_reduce_tied_grads(self) -> None:
        for (
            parameter,
            process_group,
            pipe_parallel_ranks,
        ) in self.tied_layer_index.local_parameters_and_process_groups():
            assert parameter.grad is not None
            if len(pipe_parallel_ranks) == 1:
                continue
            allreduce_tensor_in_float32(parameter.grad, process_group=process_group)  # type: ignore[arg-type]

        if self.topology.config.model_parallel_size > 1:
            for layer in self._layers:
                for parameter in layer.parameters():
                    if parameter.core_parameter_meta.tied_grad_on_model_parallel:  # type: ignore[attr-defined]
                        assert parameter.grad is not None
                        allreduce_tensor_in_float32(
                            parameter.grad,
                            process_group=self.topology.model_parallel_group,
                        )

    def reset_activation_shape(self) -> None:
        """Resets communication activation shape.

        This needs to be called simultaneously on all ranks.
        """
        self.communicator_in.reset_communication_meta()
        self.communicator_out.reset_communication_meta()
        if self.topology.config.model_parallel_size > 1:
            if self.topology.is_first_pipe_parallel_rank:
                assert self.communicator_loss_in is not None
                self.communicator_loss_in.reset_communication_meta()
            if self.topology.is_last_pipe_parallel_rank:
                assert self.communicator_loss_out is not None
                self.communicator_loss_out.reset_communication_meta()
