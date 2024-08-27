import collections
import copy
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Union

import torch

from scaling.core.logging import logger

from ..nn.parameter_meta import CoreParameterMeta
from ..topology import Topology
from ..utils.param_merge import (
    merge_parameter,
    split_parameter,
)
from .allreduce import allreduce_no_retain
from .base import BaseOptimizer, BaseOptimizerState, OptimizerStepOutput
from .loss_scaler import LossScaler, LossScalerState
from .optimizer_config import OptimizerConfig
from .parameter_group import (
    NCCL_START_ALIGNMENT_FACTOR,
    OptimizerParamGroup,
    ParameterGroupState,
    flatten_dense_tensors_aligned,
    get_data_parallel_partitions,
)


class OptimizerState(BaseOptimizerState):
    step_index: int
    parameter_groups: list[ParameterGroupState]
    optimizer: dict[str, Any]
    loss_scaler: LossScalerState


class Optimizer(BaseOptimizer):
    def __init__(
        self,
        config: OptimizerConfig,
        parameter_groups: List[OptimizerParamGroup],
        topology: Topology,
    ) -> None:
        """
        Wrapper around a torch Optimizer taking care of parallelization
        """
        self.config = config
        self.parameter_groups = parameter_groups
        self.topology = topology

        self._assert_no_parameter_duplicates()

        for parameter_group in self.parameter_groups:
            parameter_group.initialize(topology=topology, zero=self.config.zero)

        if self.config.method == "adamw":
            self.optimizer = torch.optim.AdamW(
                [p.parameter_dict for p in self.parameter_groups],
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
            self._slot_variable_names = ["exp_avg", "exp_avg_sq"]
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}.")

        self.step_index: int = 0

        self.loss_scaler = LossScaler(config=config.loss_scaler, parameter_groups=self.parameter_groups)

    def _assert_no_parameter_duplicates(self) -> None:
        counter: collections.Counter = collections.Counter()
        id_to_name = dict()
        for parameter_group in self.parameter_groups:
            for name, parameter in zip(
                parameter_group.parameter_names,
                parameter_group.parameters_original,
            ):
                counter[id(parameter)] += 1
                id_to_name[id(parameter)] = name

        parameters_occurring_more_than_once = [id_to_name[id] for id, count in counter.items() if count > 1]
        assert len(parameters_occurring_more_than_once) == 0, (
            f"parameters occurring more than once: {parameters_occurring_more_than_once}; "
            f"careful layer indices are relative to pipeline rank, "
            f"this is pipe_parallel_rank {self.topology.pipe_parallel_rank}"
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero half precision parameter grads.
        """
        for parameter_group in self.parameter_groups:
            parameter_group.zero_grad(set_to_none=set_to_none)

    def backward(self, loss: torch.Tensor) -> None:
        """
        execute backward pass on loss and potentially add loss scaling
        """
        # scale for gradient accumulation steps
        loss = loss.float()
        if self.topology.config.gradient_accumulation_steps > 1:
            loss = loss / self.topology.config.gradient_accumulation_steps

        loss = self.loss_scaler.scale_loss(loss=loss)
        loss.backward()

    def step(self) -> OptimizerStepOutput:
        """
        Do a model step, this optimizes model parameters and zeros gradients
        """

        # remember the last step taken
        self.step_index += 1

        # give dummy parameters a grad
        for parameter_group in self.parameter_groups:
            parameter_group.set_dummy_grad()

        # apply loss scaler and potentially skip step
        loss_scaler_output = self.loss_scaler.step()
        if loss_scaler_output.overflow is not None and loss_scaler_output.overflow:
            logger.warning("loss scaler encountered overflow, skipping step")
            self.zero_grad(set_to_none=True)
            return OptimizerStepOutput(
                global_grad_norm=None,
                global_grad_norm_clipped=None,
                learning_rates=None,
                overflow=loss_scaler_output.overflow,
                no_overflow_steps=loss_scaler_output.no_overflow_steps,
                current_loss_scale=loss_scaler_output.current_loss_scale,
                debug_dict=None,
            )

        self.allreduce_sequence_parallel_gradients()

        # Allreduce gradients over dp copies
        # this assumes that the prequel moved gradients to the full precision to be optimized copies
        # in order not to create redundant copies in full precision, zero reduces grads on the original parameters
        # communication will happen in full precision in case of bfloat
        if self.config.zero:
            self.allreduce_gradients(optimized_grads=False)
            global_grad_norm = self.scale_grads_and_get_global_grad_norm(optimized_grads=False)

        if self.config.debug_log:
            assert self.config.zero
            debug_dict = dict()
            for param_group in self.parameter_groups:
                for param in param_group.parameters_original:
                    scaling_parameter: CoreParameterMeta = param.core_parameter_meta  # type: ignore
                    param_name = f"{scaling_parameter.parameter_name}-layer-" f"{scaling_parameter.layer_index}"
                    if param.grad is not None:
                        debug_dict[f"debug/{param_name}-grad-norm"] = float(param.grad.norm().item())
                    debug_dict[f"debug/{param_name}-norm"] = float(param.norm().item())
                    debug_dict[f"debug/{param_name}-pct-zero"] = 1.0 - float(
                        (param.count_nonzero() / param.numel()).item()
                    )

        else:
            debug_dict = None

        # Step in learning rate scheduler
        # This will be setting the learning rate for the current step and
        # move gradients to full precision parameter copies
        # for zero this moves gradients the local copies
        for param_group in self.parameter_groups:
            param_group.step_prequel(step_index=self.step_index)

        # Allreduce gradients over dp copies
        # this assumes that the prequel moved gradients to the full precision to be optimized copies
        if not self.config.zero:
            self.allreduce_gradients(optimized_grads=True)
            global_grad_norm = self.scale_grads_and_get_global_grad_norm(optimized_grads=True)

        # Clip gradients
        self.clip_gradients(global_grad_norm=global_grad_norm)
        global_grad_norm_clipped = None  # cannot be done with gradients being released immediately

        # step in actual parameters
        self.optimizer.step()

        # This will move optimized values to half precision parameter copies
        for param_group in self.parameter_groups:
            param_group.step_sequel(
                topology=self.topology,
                numel_per_bucket=self.config.allreduce_bucket_size,
            )

        # FP32 grad should never exist outside of the step function
        # For speed, set model fp16 grad to None by default
        self.zero_grad(set_to_none=True)

        # collect learning rates
        learning_rates = dict()
        for param_group_index, param_group in enumerate(self.parameter_groups):
            name = param_group.config.name
            if name is None:
                name = f"parameter_group_{param_group_index}"
            learning_rates[name] = param_group.get_learning_rate()

        return OptimizerStepOutput(
            global_grad_norm=global_grad_norm,
            global_grad_norm_clipped=global_grad_norm_clipped,
            learning_rates=learning_rates,
            overflow=loss_scaler_output.overflow,
            no_overflow_steps=loss_scaler_output.no_overflow_steps,
            current_loss_scale=loss_scaler_output.current_loss_scale,
            debug_dict=debug_dict,
        )

    def scale_grads_and_get_global_grad_norm(
        self,
        optimized_grads: bool,
    ) -> float:
        # aggregate to global grad norm for all groups
        # this assumes that the param_group_grad_norm_squared aggregates over the full data parallel copy

        loss_scale = 1.0
        if self.loss_scaler.config.enable:
            loss_scale = self.loss_scaler._current_scale

        param_group_grad_norms_squared = list()
        for param_group in self.parameter_groups:
            param_group_grad_norm_squared = param_group.scale_grads_and_get_grad_norm_squared(
                topology=self.topology,
                optimized_grads=optimized_grads,
                loss_scale=loss_scale,
            )
            param_group_grad_norms_squared.append(param_group_grad_norm_squared)
        global_grad_norm = math.sqrt(sum(param_group_grad_norms_squared))

        return global_grad_norm

    def allreduce_gradients(self, optimized_grads: bool) -> None:
        assert self.topology.config.data_parallel_size is not None

        if self.topology.config.data_parallel_size == 1:
            return

        gradients = list()
        for parameter_group in self.parameter_groups:
            if optimized_grads:
                gradients.extend(parameter_group.get_optimized_grads())
            else:
                gradients.extend(parameter_group.get_original_grads())

        allreduce_no_retain(
            bucket=gradients,
            data_parallel_group=self.topology.data_parallel_group,
            data_parallel_size=self.topology.config.data_parallel_size,
            numel_per_bucket=self.config.allreduce_bucket_size,
        )

    def allreduce_sequence_parallel_gradients(self) -> None:
        if not self.topology.config.sequence_parallel or self.topology.config.model_parallel_size == 1:
            return

        for group in self.parameter_groups:
            for n, p in zip(group.parameter_names, group.parameters_original):
                if "norm" in n:  # TODO: This is a hack only meant for transformer training
                    torch.distributed.all_reduce(p.grad, group=self.topology.model_parallel_group)

    def clip_gradients(self, global_grad_norm: float) -> bool:
        # Do not execute if no gradient clipping is defined
        if self.config.gradient_clipping == 0.0:
            return False

        # Do not execute if the global grad norm is small enough
        if global_grad_norm < self.config.gradient_clipping:
            return False

        # actually clip the grads
        scale_factor = self.config.gradient_clipping / global_grad_norm

        # TODO needed??
        # if self.loss_scaler.config.enable:
        #     scale_factor *= self.loss_scaler._current_scale
        for param_group in self.parameter_groups:
            param_group.scale_grads(scale_factor)

        return True

    def refresh_optimizer_after_model_change(self) -> None:
        """
        Update the full precision parameter copies from the half precision copies
        the optimizer runs only on fp32 parameters
        """
        for parameter_group in self.parameter_groups:
            parameter_group.refresh_optimized_params(topology=self.topology)

    def log_state(self) -> None:
        """
        Log useful information for debugging and overall information
        """
        for param_group in self.parameter_groups:
            param_group.log_state()

    def state_dict(self) -> OptimizerState:
        """
        Get a state_dict fully representing the optimizer state
        A load of such state dict fully restores the state of the optimizer.
        """
        return {
            "step_index": self.step_index,
            "parameter_groups": [pg.state_dict() for pg in self.parameter_groups],
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
        }

    def _get_ordered_parameter_metas_and_layers(
        self,
    ) -> tuple[list[CoreParameterMeta], set[int]]:
        """
        collect parameter metadata
        assumptions are
          - torch optimizer honors the parameter order
          - we can use the param index
          - parameter order and index is the same across model parallel ranks

        """
        parameter_metas: List[CoreParameterMeta] = list()
        layer_indices = set()
        for parameter_group in self.parameter_groups:
            for parameter in parameter_group.parameters_original:
                core_parameter_meta: CoreParameterMeta = (
                    parameter.core_parameter_meta  # type: ignore
                )
                parameter_metas.append(core_parameter_meta)
                layer_indices.add(core_parameter_meta.layer_index)
                if core_parameter_meta.is_tied:
                    for layer_index in core_parameter_meta.tied_layer_indices:
                        layer_indices.add(layer_index)

        return parameter_metas, layer_indices  # type: ignore[return-value]

    def save_checkpoint(self, directory: Union[Path, str]) -> None:
        """
        Save the optimizer state to a directory.
        Assumption is that there are no name collisions of files.
        """

        if self.topology.data_parallel_rank != 0 and not self.config.zero:
            return

        logger.info("starting optimizer checkpoint save")

        # load metadata
        parameter_metas, layer_indices = self._get_ordered_parameter_metas_and_layers()

        # get local state dict
        state_dict_local = self.state_dict()

        if self.config.zero and self.config.zero_save_static:
            directory = Path(directory)
            optimizer_state_file_name = (
                f"optimizer_state_static_"
                f"mp_{self.topology.model_parallel_rank}_"
                f"pp_{self.topology.pipe_parallel_rank}_"
                f"dp_{self.topology.data_parallel_rank}.pt"
            )
            torch.save(state_dict_local, str(directory / optimizer_state_file_name))
            logger.info("saved static optimizer checkpoint")
            return

        # initialize merged state dict and copy components that are constant for model parallel
        # one merged state dict is initialized for each model layer
        # duplicated states are saved to all layer indices because we might load only one of them later

        for layer_index in layer_indices:
            state_dict_for_layer: dict[str, Any] = {
                "step_index": state_dict_local["step_index"],
                "loss_scaler": state_dict_local["loss_scaler"],
                "parameters": {},
                "optimizer_param_groups": [],
            }

            # initialize to be saved parameter groups for all layers
            for parameter_group_local, optimizer_parameter_group_local in zip(
                state_dict_local["parameter_groups"],
                state_dict_local["optimizer"]["param_groups"],
            ):
                optimizer_param_group = copy.deepcopy(optimizer_parameter_group_local)
                optimizer_param_group["params"] = []  # this will not be valid in the checkpoint
                state_dict_for_layer["optimizer_param_groups"].append(optimizer_param_group)

            # save all parameters to state dict
            for (
                step,
                core_parameter_meta_local,
                parameter_full_local,
                local_optimizer_state,
            ) in self.iterate_parameters_for_saving(
                state_dict_local,
                parameter_metas,
                layer_index,
            ):
                # merge the parameter
                merged_parameter_full = merge_parameter(
                    parameter=parameter_full_local,
                    core_parameter_meta=core_parameter_meta_local,
                    topology=self.topology,
                )

                # create metadata for merged parameter
                core_parameter_meta = CoreParameterMeta(
                    local_shape=tuple(merged_parameter_full.shape),
                    is_model_parallel=core_parameter_meta_local.is_model_parallel,
                    model_parallel_dimension=core_parameter_meta_local.model_parallel_dimension,
                    layer_index=core_parameter_meta_local.layer_index,
                    parameter_name=core_parameter_meta_local.parameter_name,
                    layer_class_name=core_parameter_meta_local.layer_class_name,
                    is_tied=core_parameter_meta_local.is_tied,
                    tied_layer_indices=core_parameter_meta_local.tied_layer_indices,
                )
                assert core_parameter_meta_local.key == core_parameter_meta.key, "key changed after merge"

                # collect the parameters optimizer state
                parameter_optimizer_state: dict[str, Any] = dict()
                parameter_optimizer_state["step"] = step
                for key in self._slot_variable_names:
                    parameter_optimizer_state[key] = merge_parameter(
                        local_optimizer_state[key],
                        core_parameter_meta=core_parameter_meta,
                        topology=self.topology,
                    )

                # record parameter in state dict
                assert core_parameter_meta.layer_index is not None
                all_layer_indices = {core_parameter_meta.layer_index}
                if core_parameter_meta.is_tied:
                    all_layer_indices = all_layer_indices.union(core_parameter_meta.tied_layer_indices)
                if layer_index in all_layer_indices:
                    state_dict_for_layer["parameters"][core_parameter_meta.key] = {
                        "parameter": merged_parameter_full,
                        "meta": core_parameter_meta.state_dict(),
                        "optimizer_state": parameter_optimizer_state,
                    }

            # collect optimizer states and merge parameters
            # this changes the structure of the optimizer state dict by breaking the reference to the local count

            if self.topology.model_parallel_rank == 0 and self.topology.data_parallel_rank == 0:
                directory = Path(directory)
                torch.save(
                    state_dict_for_layer,
                    str(directory / f"optimizer_state_layer_{layer_index}.pt"),
                )

        logger.info("saved optimizer checkpoint")

    def iterate_parameters_for_saving(
        self,
        state_dict_local: OptimizerState,
        parameter_metas: list[CoreParameterMeta],
        layer_index: int,
    ) -> Generator[tuple[int, CoreParameterMeta, torch.Tensor, dict[str, Any]], None, None]:
        global_parameter_index = -1
        for parameter_group_local, optimizer_parameter_group_local in zip(
            state_dict_local["parameter_groups"],
            state_dict_local["optimizer"]["param_groups"],
        ):
            for (
                parameter_name_local,
                parameter_meta_local,
                parameter_full_local,
                parameter_coordinates,
            ) in zip(
                parameter_group_local["parameter_names"],
                parameter_group_local["parameter_metas"],
                parameter_group_local["parameters_optimized"],
                parameter_group_local["parameter_coordinates"],
            ):
                # increment global parameter count as used in adam
                global_parameter_index += 1
                if not self.config.zero:
                    assert global_parameter_index in optimizer_parameter_group_local["params"]

                # reinitialize parameter meta from state dict
                core_parameter_meta_local = CoreParameterMeta.from_state_dict(parameter_meta_local)
                all_layer_indices = set([core_parameter_meta_local.layer_index])
                if core_parameter_meta_local.is_tied:
                    all_layer_indices = all_layer_indices.union(core_parameter_meta_local.tied_layer_indices)
                if layer_index not in all_layer_indices:
                    continue

                assert parameter_metas[global_parameter_index] == core_parameter_meta_local
                if parameter_name_local is not None and core_parameter_meta_local.parameter_name is not None:
                    assert parameter_name_local == core_parameter_meta_local.parameter_name, (
                        f"inconsistent parameter naming {parameter_name_local} "
                        f"vs. {core_parameter_meta_local.parameter_name}"
                    )

                local_parameter_optimizer_state = state_dict_local["optimizer"]["state"][
                    (optimizer_parameter_group_local["params"][0] if self.config.zero else global_parameter_index)
                ]
                step = local_parameter_optimizer_state["step"]  # constant

                if self.config.zero:
                    # collect parameter from data parallel copies
                    parameters_optimized_flat = parameter_group_local["parameters_optimized_flat"]
                    assert parameter_coordinates is not None
                    assert parameters_optimized_flat is not None
                    start, end, offset, shape = parameter_coordinates[self.topology.data_parallel_rank]
                    assert (
                        parameters_optimized_flat.dtype == torch.float32
                    ), "this assert is for paranoia reasons, optimized parameters should always be in float32"

                    parameter_comm_tensor = torch.zeros(
                        math.prod(shape),
                        dtype=parameters_optimized_flat.dtype,
                        device=parameters_optimized_flat.device,
                    )
                    optimizer_state_comm_tensors = dict()
                    for key in self._slot_variable_names:
                        optimizer_state_comm_tensors[key] = torch.zeros(
                            math.prod(shape),
                            dtype=parameters_optimized_flat.dtype,
                            device=parameters_optimized_flat.device,
                        )

                    if all([start is not None, end is not None, offset is not None]):
                        local_full = parameters_optimized_flat[start:end]
                        parameter_comm_tensor.data[offset : offset + local_full.numel()].copy_(local_full.data)

                        for key, comm_tensor in optimizer_state_comm_tensors.items():
                            local_state = local_parameter_optimizer_state[key][start:end]
                            comm_tensor.data[offset : offset + local_state.numel()].copy_(local_state.data)

                    for tensor in [parameter_comm_tensor] + list(optimizer_state_comm_tensors.values()):
                        torch.distributed.all_reduce(
                            tensor,
                            op=torch.distributed.ReduceOp.SUM,
                            group=self.topology.data_parallel_group,
                        )

                    parameter_full_local = parameter_comm_tensor.reshape(shape)
                    optimizer_states_full_local = {
                        key: comm_tensor.reshape(shape) for key, comm_tensor in optimizer_state_comm_tensors.items()
                    }

                else:
                    optimizer_states_full_local = {
                        key: local_parameter_optimizer_state[key] for key in self._slot_variable_names
                    }

                assert parameter_full_local is not None
                for key in self._slot_variable_names:
                    assert optimizer_states_full_local[key].shape == parameter_full_local.shape

                yield step, core_parameter_meta_local, parameter_full_local, optimizer_states_full_local

    def load_checkpoint(self, directory: Path | str) -> None:
        """
        Load the state into an already initialized optimizer
        """

        if self.config.zero and self.config.zero_save_static:
            directory = Path(directory)
            optimizer_state_file_name = (
                f"optimizer_state_static_"
                f"mp_{self.topology.model_parallel_rank}_"
                f"pp_{self.topology.pipe_parallel_rank}_"
                f"dp_{self.topology.data_parallel_rank}.pt"
            )
            checkpoint = torch.load(
                str(directory / optimizer_state_file_name),
            )

            # load constant state
            self.step_index = checkpoint["step_index"]  # constant, does not matter which layer to use
            self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])  # constant, does not matter which layer to use
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for parameter_group_index, (
                parameter_group,
                torch_optimizer_parameter_group,
                loaded_state_dict,
            ) in enumerate(
                zip(
                    self.parameter_groups,
                    self.optimizer.param_groups,
                    checkpoint["parameter_groups"],
                )
            ):
                parameter_group.load_state_dict(loaded_state_dict, zero_load_static=True)

                # torch optimizer deep copies
                parameter_group.parameter_dict = torch_optimizer_parameter_group

            logger.info("loaded static optimizer checkpoint")
            return

        # get currently initialized local parameters
        parameter_metas, layer_indices_set = self._get_ordered_parameter_metas_and_layers()
        layer_indices = list(layer_indices_set)

        # load all local layers
        directory = Path(directory)
        state_dict_by_layer = dict()
        for layer_index in layer_indices:
            state_dict_file = directory / f"optimizer_state_layer_{layer_index}.pt"
            state_dict_by_layer[layer_index] = torch.load(str(state_dict_file))

        # make sure all parameters that should be constant are constant
        if len(layer_indices) > 1:
            for layer_index_compare in layer_indices[1:]:
                assert (
                    state_dict_by_layer[layer_indices[0]]["step_index"]
                    == state_dict_by_layer[layer_index_compare]["step_index"]
                )
                assert (
                    state_dict_by_layer[layer_indices[0]]["loss_scaler"]
                    == state_dict_by_layer[layer_index_compare]["loss_scaler"]
                )
                assert (
                    state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"]
                    == state_dict_by_layer[layer_index_compare]["optimizer_param_groups"]
                )

        # load constant state
        # constant, does not matter which layer to use
        self.step_index = state_dict_by_layer[layer_indices[0]]["step_index"]
        self.loss_scaler.load_state_dict(state_dict_by_layer[layer_indices[0]]["loss_scaler"])

        # localize parameters and collect by key
        parameters = dict()
        for _layer_index, state_dict_for_layer in state_dict_by_layer.items():
            for _param_key, parameter_state_dict in state_dict_for_layer["parameters"].items():
                aleph_alpha_parameter_meta = CoreParameterMeta.from_state_dict(parameter_state_dict["meta"])
                if aleph_alpha_parameter_meta.is_model_parallel:
                    parameter_state_dict["parameter"] = split_parameter(
                        parameter=parameter_state_dict["parameter"],
                        core_parameter_meta=aleph_alpha_parameter_meta,
                        topology=self.topology,
                    )
                    for key in self._slot_variable_names:
                        parameter_state_dict["optimizer_state"][key] = split_parameter(
                            parameter=parameter_state_dict["optimizer_state"][key],
                            core_parameter_meta=aleph_alpha_parameter_meta,
                            topology=self.topology,
                        )

                for possible_key in aleph_alpha_parameter_meta.possible_keys():
                    parameters[possible_key] = parameter_state_dict

        # collect and load optimizer state
        optimizer_state_dict: Dict = dict()
        optimizer_state_dict_current = self.optimizer.state_dict()
        optimizer_state_dict["state"] = {}
        optimizer_state_dict["param_groups"] = list()
        assert self.topology.config.data_parallel_size is not None
        assert len(optimizer_state_dict_current["param_groups"]) == len(
            state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"]
        )
        for param_group_current, param_group_loaded in zip(
            optimizer_state_dict_current["param_groups"],
            state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"],
        ):
            # get the pointers to the current parameters
            param_group_loaded["params"] = copy.deepcopy(param_group_current["params"])

            # set the param groups state
            optimizer_state_dict["param_groups"].append(param_group_loaded)

            # set the parameter states
            if self.config.zero:
                for parameter_group_index, parameter_group in enumerate(self.parameter_groups):
                    optimizer_state_dict["state"][parameter_group_index] = {
                        "step": parameters[parameter_group.parameter_metas[0].key]["optimizer_state"]["step"]
                    }
                    for key in self._slot_variable_names:
                        opt_state_list = [
                            parameters[parameter_meta.key]["optimizer_state"][key]
                            for parameter_meta in parameter_group.parameter_metas
                        ]
                        opt_state_tensor, _zero_padding = flatten_dense_tensors_aligned(
                            opt_state_list,
                            NCCL_START_ALIGNMENT_FACTOR * self.topology.config.data_parallel_size,
                        )
                        opt_state = get_data_parallel_partitions(opt_state_tensor, topology=self.topology)[
                            self.topology.data_parallel_rank
                        ]

                        optimizer_state_dict["state"][parameter_group_index][key] = opt_state

            else:
                for param_index in param_group_loaded["params"]:
                    assert param_index not in optimizer_state_dict["state"], "duplicate param index"
                    optimizer_state_dict["state"][param_index] = parameters[parameter_metas[param_index].key][
                        "optimizer_state"
                    ]

        self.optimizer.load_state_dict(optimizer_state_dict)

        for parameter_group_index, (
            parameter_group,
            torch_optimizer_parameter_group,
        ) in enumerate(
            zip(
                self.parameter_groups,
                self.optimizer.param_groups,
            )
        ):
            current_state_dict = parameter_group.state_dict()
            loaded_state_dict = dict()
            loaded_state_dict["parameter_names"] = copy.deepcopy(current_state_dict["parameter_names"])
            loaded_state_dict["parameter_metas"] = copy.deepcopy(current_state_dict["parameter_metas"])

            if self.config.zero:
                parameter_list = [
                    parameters[parameter_meta.key]["parameter"] for parameter_meta in parameter_group.parameter_metas
                ]
                parameter_tensor, _zero_padding = flatten_dense_tensors_aligned(
                    parameter_list,
                    NCCL_START_ALIGNMENT_FACTOR * self.topology.config.data_parallel_size,
                )
                parameter = get_data_parallel_partitions(parameter_tensor, topology=self.topology)[
                    self.topology.data_parallel_rank
                ]
                parameters_optimized = [parameter]
            else:
                parameters_optimized = list()
                for parameter_name, parameter_meta_state_dict in zip(
                    current_state_dict["parameter_names"],
                    current_state_dict["parameter_metas"],
                ):
                    aleph_alpha_parameter_meta = CoreParameterMeta.from_state_dict(parameter_meta_state_dict)
                    assert parameter_name == aleph_alpha_parameter_meta.parameter_name
                    parameters_optimized.append(parameters[aleph_alpha_parameter_meta.key]["parameter"])
            loaded_state_dict["parameters_optimized"] = parameters_optimized
            parameter_group.load_state_dict(loaded_state_dict)

            # torch optimizer deep copies
            parameter_group.parameter_dict = torch_optimizer_parameter_group

        logger.info("loaded optimizer checkpoint")
