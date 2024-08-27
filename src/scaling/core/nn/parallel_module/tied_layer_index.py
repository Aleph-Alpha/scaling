import collections
from typing import Optional, Sequence

import torch

from ...topology import Topology
from .layer_spec import LayerSpec, TiedLayerSpec
from .pipeline_partitioning import PipePartitionCoordinates


class TiedLayerInformation:
    def __init__(self) -> None:
        self.is_local: bool = False
        self.layer_indices: set[int] = set()
        self.pipe_parallel_ranks: set[int] = set()
        self.tied_weight_attributes: set[str] = set()
        self.process_group: Optional[torch.distributed.ProcessGroup] = None
        self.global_ranks: Optional[list[int]] = None

    def add(
        self,
        layer_index: int,
        pipe_parallel_rank: int,
        is_local: bool,
        tied_weight_attributes: Sequence[str],
    ) -> None:
        self.layer_indices.add(layer_index)
        self.pipe_parallel_ranks.add(pipe_parallel_rank)
        self.is_local = self.is_local or is_local

        for tied_weight_attribute in tied_weight_attributes:
            self.tied_weight_attributes.add(tied_weight_attribute)

    def build_process_groups(self, topology: Topology) -> None:
        assert topology.config.data_parallel_size is not None

        for data_parallel_rank in range(topology.config.data_parallel_size):
            for model_parallel_rank in range(topology.config.model_parallel_size):
                # collect all global ranks for combinations of data parallel and model parallel
                global_ranks = list()
                for pipe_parallel_rank in self.pipe_parallel_ranks:
                    global_rank = topology.get_global_rank(
                        data_parallel_rank=data_parallel_rank,
                        model_parallel_rank=model_parallel_rank,
                        pipe_parallel_rank=pipe_parallel_rank,
                    )
                    global_ranks.append(global_rank)
                global_ranks = sorted(global_ranks)

                # build group
                if len(global_ranks) > 1 and topology.is_distributed_initialized:
                    group = torch.distributed.new_group(global_ranks)
                else:
                    group = None
                if topology.config.global_rank in global_ranks:
                    self.process_group = group
                    self.global_ranks = global_ranks


class TiedModule:
    def __init__(
        self,
        module: torch.nn.Module,
        is_main: bool,
        is_local_main: bool,
        tied_weight_attributes: list[str],
    ):
        self.module = module
        self.is_main = is_main
        self.is_local_main = is_local_main
        self.tied_weight_attributes = set(tied_weight_attributes)


class TiedLayerIndex:
    def __init__(
        self,
        pipe_partition_coordinates: list[PipePartitionCoordinates],
        layer_specs: list[LayerSpec],
        topology: Topology,
    ) -> None:
        # get own pipe coordinates
        self._pipe_partition_coordinate = pipe_partition_coordinates[topology.pipe_parallel_rank]

        # map layer to pipe_parallel_rank
        self.layer_index_to_pipe_parallel_rank = dict()
        for pipe_parallel_rank, pipe_partition_coordinate in enumerate(pipe_partition_coordinates):
            for layer_index in range(pipe_partition_coordinate.start, pipe_partition_coordinate.end):
                self.layer_index_to_pipe_parallel_rank[layer_index] = pipe_parallel_rank

        # collect tied information by key
        self.tied_information_by_key: dict[str, TiedLayerInformation] = collections.defaultdict(TiedLayerInformation)
        for layer_index, layer_spec in enumerate(layer_specs):
            # only work on tied layer specs
            if not isinstance(layer_spec, TiedLayerSpec):
                continue

            is_local = self._pipe_partition_coordinate.start <= layer_index < self._pipe_partition_coordinate.end
            self.tied_information_by_key[layer_spec.key].add(
                is_local=is_local,
                layer_index=layer_index,
                pipe_parallel_rank=self.layer_index_to_pipe_parallel_rank[layer_index],
                tied_weight_attributes=layer_spec.tied_weight_attributes,
            )

        # build the process groups
        # build of all groups have to be called for all ranks (compare topology docstrings)
        for key in sorted(self.tied_information_by_key.keys()):  # sort to be really sure about the order
            tied_information = self.tied_information_by_key[key]
            tied_information.build_process_groups(topology=topology)
            if (
                tied_information.is_local
                and isinstance(tied_information.global_ranks, list)
                and len(tied_information.global_ranks) > 1
                and topology.is_distributed_initialized
            ):
                assert (
                    tied_information.process_group is not None
                ), "local tied layers must have a process group if more than one rank is involved"

        # collect tied modules
        tied_keys_counter: dict[str, int] = collections.Counter()
        tied_keys_counter_local: dict[str, int] = collections.Counter()
        self.tied_modules_local_main_by_key = dict()
        self.tied_parameters_by_key = dict()
        self.tied_modules_by_layer_index: dict[int, TiedModule] = dict()
        for layer_index, layer_spec in enumerate(layer_specs):
            # only work on tied layer specs
            if not isinstance(layer_spec, TiedLayerSpec):
                continue

            # derive information for local pipe stage
            is_local = self._pipe_partition_coordinate.start <= layer_index < self._pipe_partition_coordinate.end
            is_main = tied_keys_counter[layer_spec.key] == 0
            is_local_main = tied_keys_counter_local[layer_spec.key] == 0

            # increment counts
            tied_keys_counter[layer_spec.key] += 1
            if is_local:
                tied_keys_counter_local[layer_spec.key] += 1

            # instantiate module
            if is_local:
                # build
                module = layer_spec.initialize()
                tied_module = TiedModule(
                    module=module,
                    is_main=is_main,
                    is_local_main=is_local_main,
                    tied_weight_attributes=layer_spec.tied_weight_attributes,
                )

                if is_local_main:
                    self.tied_modules_local_main_by_key[layer_spec.key] = tied_module
                    self.tied_parameters_by_key[layer_spec.key] = {
                        n: p
                        for n, p in module.named_parameters()
                        if n in self.tied_information_by_key[layer_spec.key].tied_weight_attributes
                    }
                else:
                    module_tied_to_module_main = self.tied_modules_local_main_by_key[layer_spec.key].module
                    module_tied_to_module = tied_module.module
                    for tied_weight_attribute in layer_spec.tied_weight_attributes:
                        tied_weight_attribute_split = tied_weight_attribute.split(".")
                        tied_module_main_attr = module_tied_to_module_main
                        tied_module_attr = module_tied_to_module
                        while len(tied_weight_attribute_split) > 1:
                            tied_module_main_attr = getattr(tied_module_main_attr, tied_weight_attribute_split[0])
                            tied_module_attr = getattr(tied_module_attr, tied_weight_attribute_split[0])
                            tied_weight_attribute_split = tied_weight_attribute_split[1:]

                        setattr(
                            tied_module_attr,
                            tied_weight_attribute_split[0],
                            getattr(tied_module_main_attr, tied_weight_attribute_split[0]),
                        )
                self.tied_modules_by_layer_index[layer_index] = tied_module

    def get_module_by_layer_index(self, layer_index: int) -> torch.nn.Module:
        return self.tied_modules_by_layer_index[layer_index].module

    def get_tied_weight_attributes_by_layer_index(self, layer_index: int) -> set[str]:
        return self.tied_modules_by_layer_index[layer_index].tied_weight_attributes

    def local_parameters_and_process_groups(
        self,
    ) -> list[tuple[torch.nn.Parameter, Optional[torch.distributed.ProcessGroup], set[int]]]:
        parameter_and_process_groups = list()
        for key in sorted(list(self.tied_parameters_by_key.keys())):
            for parameter_name in sorted(list(self.tied_parameters_by_key[key].keys())):
                parameter = self.tied_parameters_by_key[key][parameter_name]
                process_group = self.tied_information_by_key[key].process_group
                pipe_parallel_ranks = self.tied_information_by_key[key].pipe_parallel_ranks
                if len(pipe_parallel_ranks) > 1:
                    assert process_group is not None
                parameter_and_process_groups.append((parameter, process_group, pipe_parallel_ranks))

        return parameter_and_process_groups

    def layer_index_is_tied_global_duplicate(self, layer_index: int) -> bool:
        is_duplicate = False
        if layer_index in self.tied_modules_by_layer_index:
            if not self.tied_modules_by_layer_index[layer_index].is_main:
                is_duplicate = True

        return is_duplicate

    def layer_index_is_tied_local_duplicate(self, layer_index: int) -> bool:
        is_duplicate = False
        if layer_index in self.tied_modules_by_layer_index:
            if not self.tied_modules_by_layer_index[layer_index].is_local_main:
                is_duplicate = True

        return is_duplicate

    def layer_index_to_tied_local_duplicate_parameter_names(self, layer_index: int) -> set[str]:
        # if layer is not tied, no parameters can be tied duplicates
        if layer_index not in self.tied_modules_by_layer_index:
            return set()

        # if layer is local main, no parameters can be tied duplicates
        if self.tied_modules_by_layer_index[layer_index].is_local_main:
            return set()

        return self.tied_modules_by_layer_index[layer_index].tied_weight_attributes
