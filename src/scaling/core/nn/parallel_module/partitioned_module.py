import re
from pathlib import Path
from typing import (
    Optional,
    Sequence,
    Union,
)

import torch

from scaling.core.logging import logger
from scaling.core.nn.parallel_module.pipeline_partitioning import (
    PipePartitionCoordinates,
    pipe_partition_balanced,
    pipe_partition_from_indices,
    pipe_partition_uniform,
)
from scaling.core.nn.parallel_module.tied_layer_index import TiedLayerIndex
from scaling.core.nn.parameter_meta import CoreParameterMeta, UMuPParameterMeta
from scaling.core.topology import PipePartitionMethod, Topology
from scaling.core.utils.param_merge import merge_parameter, split_parameter

from .layer_spec import LayerSpec, TiedLayerSpec

legacy_checkpoint_mapping = {
    "TransformerEmbeddingHead": "LuminousEmbeddingHead",
    "EmbeddingInput": "LuminousEmbeddingInput",
    "TransformerLayer": "LuminousLayer",
    "TransformerLayerIO": "LuminousLayerIO",
    "TransformerLMHead": "LuminousLMHead",
    "TransformerLMHeadTied": "LuminousLMHeadTied",
}


def key_match(key: str, list_of_patterns: list[str]) -> bool:
    """Compares a string to a list of patterns.

    Returns `True` if any of the patterns appears in `key`. This is used to match
    checkpoint keys against patterns that are allowed to be missing/unexpected.
    """
    return any(re.search(pattern, key) is not None for pattern in list_of_patterns)


class PipePartitionedModule(
    torch.nn.Module,
):
    """Base module class for splitting a module consisting of several layers onto different devices.

    Can be used in a single process scenario (mainly for inference),
    in this case the list of devices needs to be provided and every layer gets
    initialized on its assigned device.

    Can also be used in a multiprocess scenario (mainly for training),
    in which case a topology object needs to be passed and each process only initializes
    its owned partition of layers.
    """

    def __init__(
        self,
        layer_specs: list[LayerSpec],
        devices: Optional[Sequence[int]] = None,
        topology: Optional[Topology] = None,
        pipe_partition_method: Optional[PipePartitionMethod] = None,
        pipe_partition_overwrite: Optional[list[int]] = None,
    ):
        super().__init__()

        self.topology = topology
        self.devices: list[torch.device] | None = None

        assert (devices is None) ^ (topology is None), "Exactly either one of 'devices' or 'topology' must be specified"

        if devices is not None:
            self.devices = [torch.device("cuda", device_idx) for device_idx in devices]

        # Initialize and sync model.
        self._layer_specs = layer_specs
        self.topology = topology
        if self.topology is None:
            if pipe_partition_method is None:
                self.pipe_partition_method = PipePartitionMethod.UNIFORM
            else:
                self.pipe_partition_method = pipe_partition_method
            self.pipe_partition_overwrite = pipe_partition_overwrite
        else:
            assert pipe_partition_method is None, "pipe partition method should be provided through the topology"
            assert pipe_partition_overwrite is None, "pipe partition overwrite should be provided through the topology"
            self.pipe_partition_method = self.topology.config.pipe_partition_method
            self.pipe_partition_overwrite = self.topology.config.pipe_partition_overwrite

        self._initialize_layers()

    def _get_pipe_partition_coordinates(self) -> list[PipePartitionCoordinates]:
        # Make sure the topology fits the number of layer specs.
        if self.devices is not None:
            num_partitions = len(self.devices)
        elif self.topology is not None:
            num_partitions = self.topology.config.pipe_parallel_size

        if len(self._layer_specs) < num_partitions:
            raise RuntimeError(
                f"Number of layers ({len(self._layer_specs)}) is smaller"
                f"than number of pipe partitions {num_partitions}"
            )

        if self.pipe_partition_overwrite is not None:
            pipe_partition_coordinates = pipe_partition_from_indices(
                self.pipe_partition_overwrite,
                num_layers=len(self._layer_specs),
            )
        elif self.pipe_partition_method == PipePartitionMethod.UNIFORM:
            pipe_partition_coordinates = pipe_partition_uniform(
                item_count=len(self._layer_specs),
                partition_count=num_partitions,
            )
        elif self.pipe_partition_method == PipePartitionMethod.BALANCED:
            pipe_partition_coordinates = pipe_partition_balanced(
                layer_specs=self._layer_specs,
                partition_count=num_partitions,
            )
        else:
            raise NotImplementedError(f"Pipe partition method not known: {self.pipe_partition_method}")

        return pipe_partition_coordinates

    def _initialize_layers(self) -> None:
        """Initializes modules in layer specs on the basis of pipeline partitions."""

        pipe_partition_coordinates = self._get_pipe_partition_coordinates()

        for idx, pipe_partition_coordinate in enumerate(pipe_partition_coordinates):
            if (self.topology is not None and self.topology.config.global_rank == 0) or self.topology is None:
                logger.warning(
                    f"pipe_parallel_rank {idx} gets layers "
                    f"{pipe_partition_coordinate.start}:{pipe_partition_coordinate.end}"
                )
            assert (
                pipe_partition_coordinate.start < pipe_partition_coordinate.end
            ), f"no parallel module layer spec assigned to index {idx}"

        self._pipe_partition_coordinates = pipe_partition_coordinates

        if self.topology is not None:
            self._pipe_partition_coordinates = [pipe_partition_coordinates[self.topology.pipe_parallel_rank]]

        # Index tied modules.
        self.tied_layer_index: TiedLayerIndex | None
        if self.topology is not None:
            self.tied_layer_index = TiedLayerIndex(
                pipe_partition_coordinates=pipe_partition_coordinates,
                layer_specs=self._layer_specs,
                topology=self.topology,
            )
        else:
            self.tied_layer_index = None  # TODO how to handle weight tying in non parallel case?

        # Initialize layers.
        self._layers = torch.nn.ModuleList()

        if self.devices is not None:
            devices = self.devices
        elif self.topology is not None:
            devices = [self.topology.device]

        for device, pipe_partition_coordinate in zip(devices, self._pipe_partition_coordinates):
            for local_layer_index, layer_spec in enumerate(
                self._layer_specs[pipe_partition_coordinate.start : pipe_partition_coordinate.end]
            ):
                layer_index = pipe_partition_coordinate.start + local_layer_index
                if isinstance(layer_spec, TiedLayerSpec):
                    assert (
                        self.tied_layer_index is not None
                    ), "TiedLayerSpecs are currently only supported when providing a topology"
                    _layer = self.tied_layer_index.get_module_by_layer_index(layer_index=layer_index)
                    tied_weight_attributes = self.tied_layer_index.get_tied_weight_attributes_by_layer_index(
                        layer_index=layer_index
                    )
                    layer_is_tied = True
                else:
                    _layer = layer_spec.initialize(device=device)
                    tied_weight_attributes = set()
                    layer_is_tied = False

                # Add meta.
                for parameter_name, parameter in list(_layer.named_parameters()) + list(_layer.named_buffers()):
                    parameter_is_tied = layer_is_tied and parameter_name in tied_weight_attributes
                    if hasattr(parameter, "core_parameter_meta"):
                        parameter.core_parameter_meta.set(
                            layer_index=layer_index,
                            parameter_name=parameter_name,
                            layer_class_name=_layer.__class__.__name__,
                            is_tied=parameter_is_tied,
                        )
                    else:
                        CoreParameterMeta.register_on_parameter(
                            parameter=parameter,
                            is_model_parallel=False,
                            layer_index=layer_index,
                            parameter_name=parameter_name,
                            layer_class_name=_layer.__class__.__name__,
                            is_tied=parameter_is_tied,
                        )

                self._layers.append(_layer)

    def umup_setup(
        self,
        effective_batch_size: int,
        depth: int,
        avg_sequence_length: int,
        allow_non_umup_params: bool = False,
    ) -> None:
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "umup_setup"):
                module.umup_setup(
                    effective_batch_size=effective_batch_size,
                    depth=depth,
                    avg_sequence_length=avg_sequence_length,
                )
                # We typically use the paradigm to switch to a different forward pass
                # via the _use_umup flag that gets modified by umup_setup.
                # The assert is to ensure we do not forget to switch the flag
                assert hasattr(module, "_use_umup") and module._use_umup, module.__class__
        if not allow_non_umup_params:
            for n, p in self.named_parameters():
                assert hasattr(p, "core_parameter_meta")
                assert isinstance(p.core_parameter_meta, CoreParameterMeta)
                assert isinstance(
                    p.core_parameter_meta.umup_meta, UMuPParameterMeta
                ), f"mup parameter meta not set for {n}"
                assert hasattr(
                    p.core_parameter_meta.umup_meta, "forward_multiplier"
                ), f"forward multiplier is not set for parameter {n}"
                assert hasattr(
                    p.core_parameter_meta.umup_meta, "backward_multiplier"
                ), f"backward multiplier is not set for parameter {n}"
                assert hasattr(
                    p.core_parameter_meta.umup_meta, "grad_multiplier"
                ), f"grad multiplier is not set for parameter {n}"
                assert hasattr(
                    p.core_parameter_meta.umup_meta, "lr_multiplier"
                ), f"lr multiplier is not set for parameter {n}"

    def save_checkpoint(self, dir_: Union[Path, str], separate_file_for_parameters: Optional[list[str]]) -> None:
        """Saves the model state to a directory.

        One file is saved per layer. Model parallel parameters are merged upon save.
        """
        if self.topology is not None and self.topology.data_parallel_rank != 0:
            return
        dir_ = Path(dir_)

        # Collect the state dict for each layer.
        # layer.state_dict() is not simply called in order to be able to collect model parallel parameters
        for pipe_coordinate in self._pipe_partition_coordinates:
            for local_layer_index, layer in enumerate(self._layers):
                # iterate layer by layer in order to save one file per layer
                # create one state dict for each special parameter group to be saved
                state_dicts: dict[str, dict[str, torch.Tensor]] = {"": dict()}
                if separate_file_for_parameters is not None:
                    for state_dict_parameter_name in separate_file_for_parameters:
                        state_dicts[state_dict_parameter_name] = dict()

                names_in_state_dict = list(layer.state_dict().keys())
                # named_parameters() gives the same names that would otherwise be present when calling .state_dict()
                for parameter_name, parameter in list(layer.named_parameters()) + list(layer.named_buffers()):
                    if parameter_name not in names_in_state_dict:
                        # we make sure that non-persistent buffers are excluded
                        continue
                    core_parameter_meta: CoreParameterMeta = parameter.core_parameter_meta
                    if self.topology is not None:
                        parameter_merged = merge_parameter(
                            parameter=parameter,
                            core_parameter_meta=core_parameter_meta,
                            topology=self.topology,
                        )
                    else:
                        parameter_merged = parameter

                    state_dict_name = ""
                    if separate_file_for_parameters is not None:
                        for state_dict_parameter_name in separate_file_for_parameters:
                            if state_dict_parameter_name in parameter_name:
                                state_dict_name = state_dict_parameter_name

                    state_dicts[state_dict_name][parameter_name] = parameter_merged

                if self.topology is None or self.topology.model_parallel_rank == 0:
                    for state_dict_name, state_dict in state_dicts.items():
                        model_state_layer_file_name = (
                            f"model_state_layer_"
                            f"{pipe_coordinate.start + local_layer_index}_"
                            f"{layer.__class__.__name__}"
                            f"{'' if state_dict_name == '' else '_'}{state_dict_name}.pt"
                        )
                        if not state_dict:
                            continue
                        torch.save(state_dict, str(dir_ / model_state_layer_file_name))

    def load_checkpoint(
        self,
        dir_: Path | str | list[Path | str],
        add_bias_names_if_not_exist: Optional[list[str]] = None,
        add_bias_names_if_not_exist_exceptions: Optional[list[str]] = None,
        allowed_missing_keys_in_checkpoint: Optional[list[str]] = None,
        allowed_unexpected_keys_in_checkpoint: Optional[list[str]] = None,
        ignore_keys_in_checkpoint: Optional[list[str]] = None,
    ) -> None:
        """Loads the state into an already initialized module."""

        # To ease deployment a list of source directories can be passed.
        # The directories are searched in list order for the parameter files of each layer.
        input_list = [dir_] if not isinstance(dir_, list) else dir_
        input_paths = [Path(path) for path in input_list]
        missing_dirs = list(filter(lambda p: not p.is_dir(), input_paths))
        legacy_checkpoint = _is_legacy_checkpoint(input_paths)
        if missing_dirs:
            raise RuntimeError(f"Weight set directories missing 📢🚨❗: {missing_dirs}")

        # Replace Nones with empty lists for easier handling below.
        allowed_missing_keys_in_checkpoint = allowed_missing_keys_in_checkpoint or []
        allowed_unexpected_keys_in_checkpoint = allowed_unexpected_keys_in_checkpoint or []
        ignore_keys_in_checkpoint = ignore_keys_in_checkpoint or []
        allowed_unexpected_keys_in_checkpoint.extend(ignore_keys_in_checkpoint)

        # Sets to collect unexpected/missing keys across layers.
        unexpected_keys = set()
        missing_keys = set()
        layer_index_offset = 0 if self.topology is None else self._pipe_partition_coordinates[0].start

        for local_layer_index, layer in enumerate(self._layers):
            layer_index = layer_index_offset + local_layer_index
            logger.debug(f"Loading model state for layer {layer_index}.")
            state_dict = {}
            layer_class_name = layer.__class__.__name__
            if legacy_checkpoint:
                layer_class_name = legacy_checkpoint_mapping.get(layer_class_name, layer_class_name)
            for path in input_paths:
                for state_dict_file in path.glob(f"model_state_layer_{layer_index}_{layer_class_name}*.pt"):
                    state_dict.update(torch.load(state_dict_file))
            logger.debug(f"Done loading model state for layer {layer_index}.")

            if add_bias_names_if_not_exist is not None:
                for k in list(state_dict.keys()):
                    if add_bias_names_if_not_exist_exceptions is not None:
                        if any(
                            any([k_split == e for k_split in k.split(".")])
                            for e in add_bias_names_if_not_exist_exceptions
                        ):
                            continue

                    if k.endswith(".bias"):
                        for bias_name in add_bias_names_if_not_exist:
                            if bias_name is None:
                                continue
                            if bias_name == "":
                                continue
                            new_bias_parameter = k + "_" + bias_name
                            if new_bias_parameter not in state_dict:
                                state_dict[new_bias_parameter] = state_dict[k].clone()

            # If model parallel:
            # overwrite state_dict with parallel split parameters for the current model parallel rank
            if self.topology is not None and self.topology.config.model_parallel_size > 1:
                for parameter_name, parameter in list(layer.named_parameters()) + list(layer.named_buffers()):
                    if parameter_name in state_dict and parameter.core_parameter_meta.is_model_parallel:
                        parameters_for_all_mp = state_dict[parameter_name].data
                        split_parameter_for_mp_rank = split_parameter(
                            parameter=parameters_for_all_mp,
                            core_parameter_meta=parameter.core_parameter_meta,
                            topology=self.topology,
                        )
                        state_dict[parameter_name] = split_parameter_for_mp_rank.to(self.topology.device)

            # Remove unwanted keys.
            if ignore_keys_in_checkpoint is not None:
                state_dict = {k: v for k, v in state_dict.items() if not key_match(k, ignore_keys_in_checkpoint)}

            # Load state_dict and track missing and unexpected keys.
            incompatible_keys = layer.load_state_dict(state_dict, strict=False)
            missing_keys.update(incompatible_keys.missing_keys)
            unexpected_keys.update(incompatible_keys.unexpected_keys)

        # Check whether unexpected keys were allowed.
        unexpected_keys_error, unexpected_keys_ignored = _extract_matching_keys(
            unexpected_keys, allowed_unexpected_keys_in_checkpoint
        )
        if unexpected_keys_ignored:
            logger.warning(f"Ignoring unexpected keys in checkpoint: {unexpected_keys_ignored}")
        if unexpected_keys_error:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected_keys_error}. You may add "
                f"keys to 'allowed_unexpected_keys_in_checkpoint'."
            )

        # Check whether missing keys are allowed.
        missing_keys_error, missing_keys_ignored = _extract_matching_keys(
            missing_keys, allowed_missing_keys_in_checkpoint
        )
        if missing_keys_ignored:
            logger.warning(f"Ignoring missing keys in checkpoint: {missing_keys_ignored}")
        if missing_keys_error:
            raise RuntimeError(
                f"Missing keys in checkpoint: {missing_keys_error}. You may add "
                f"keys to 'allowed_missing_keys_in_checkpoint'."
            )


def _extract_matching_keys(keys: set[str], keys_to_match: list[str]) -> tuple[set[str], set[str]]:
    keys_matching = set()
    keys_not_matching = set()
    for key in keys:
        if key_match(key, keys_to_match):
            keys_matching.add(key)
        else:
            keys_not_matching.add(key)
    return keys_not_matching, keys_matching


def _is_legacy_checkpoint(paths: Sequence[Path]) -> bool:
    for path in paths:
        for legacy_name in legacy_checkpoint_mapping.values():
            if any(path.glob(f"*{legacy_name}.pt")):
                return True
    return False
