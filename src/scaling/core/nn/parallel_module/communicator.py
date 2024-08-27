import pickle
from collections import namedtuple
from ctypes import c_uint64
from functools import partial
from typing import Any, Optional

import torch
from torch.utils._pytree import (
    SUPPORTED_NODES,
    LeafSpec,
    PyTree,
    TreeSpec,
    tree_flatten,
    tree_unflatten,
)

from scaling.core.logging import logger
from scaling.core.topology import Topology

GradOutput = namedtuple("GradOutput", "tensors,grad_tensors")


class CommunicationMetaBase:
    def __init__(
        self,
        is_tensor: bool,
        tensor_shape: torch.Size,
        tensor_dtype: torch.TensorType,
        requires_grad: bool,
    ):
        self.is_tensor = is_tensor
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.requires_grad = requires_grad

    def compare(self, other: "CommunicationMetaBase") -> bool:
        if self.is_tensor != other.is_tensor:
            return False

        if self.tensor_shape != other.tensor_shape:
            return False

        if self.tensor_dtype != other.tensor_dtype:
            return False

        if self.requires_grad != other.requires_grad:
            return False

        return True


def _dump_to_pickle_tensor(x: Any) -> tuple[torch.Tensor, int]:
    dump: list[int] = list(pickle.dumps(x))
    dump_size: int = len(dump)

    tensor_dump: torch.Tensor = torch.tensor(dump, dtype=torch.uint8)

    return tensor_dump, dump_size


def dump_to_pickle_tensor(x: Any, max_buffer_size: Optional[int] = None) -> torch.Tensor:
    dump: list[int] = list(pickle.dumps(x))
    dump_size: int = len(dump)
    dump_size_c: c_uint64 = c_uint64(dump_size)

    if max_buffer_size is None:
        dump_pad_size = dump_size

    elif (dump_size + 8) <= max_buffer_size:
        dump_pad_size = max_buffer_size - dump_size - 8

    else:
        dump_pad_size = dump_size

    dump = list(bytes(dump_size_c)) + dump + [0] * dump_pad_size  # type: ignore
    dump_tensor = torch.tensor(dump, dtype=torch.uint8)

    return dump_tensor


def map_loads_from_pickle_tensor_fn(t: torch.Tensor, meta: CommunicationMetaBase) -> Any:
    if meta.is_tensor:
        return t

    dumps: list[int] = t.tolist()

    size_ints: list[int] = dumps[:8]
    size_bytes: bytes = bytes(size_ints)
    size = int.from_bytes(size_bytes, byteorder="little")

    dumps = dumps[8:][:size]
    dumps_b: bytes = bytes(dumps)

    return pickle.loads(dumps_b)


def map_to_meta_and_tensor_fn(
    device: torch.device, x: Any, max_buffer_size: Optional[int] = None
) -> tuple[CommunicationMetaBase, torch.Tensor]:
    if torch.is_tensor(x):
        meta = CommunicationMetaBase(
            is_tensor=True,
            tensor_shape=x.size(),
            tensor_dtype=x.dtype,
            requires_grad=x.requires_grad,
        )

        return meta, x

    else:
        dump_tensor = dump_to_pickle_tensor(x, max_buffer_size=max_buffer_size)
        dump_tensor = dump_tensor.to(device)

        meta = CommunicationMetaBase(
            is_tensor=False,
            tensor_shape=dump_tensor.size(),
            tensor_dtype=torch.uint8,  # type: ignore
            requires_grad=False,
        )

        return meta, dump_tensor


def map_lower_to_tensor_fn(device: torch.device, x: Any, meta: CommunicationMetaBase) -> torch.Tensor:
    if torch.is_tensor(x):
        assert x.size() == meta.tensor_shape, (
            f"Try to communicate a tensor with a different size than saved in the meta."
            f" Tensor size is {x.size()} but meta size is: {meta.tensor_shape}."
            f" Try to reset the 'communication_meta'."
        )

        assert x.dtype == meta.tensor_dtype, (
            f"Try to communicate a tensor with a different dtype than saved in the meta."
            f" Tensor dtype is {x.dtype} but meta dtype is: {meta.tensor_dtype}."
            f" Try to reset the 'communication_meta'."
        )

        return x

    else:
        dump_tensor = dump_to_pickle_tensor(x, max_buffer_size=meta.tensor_shape[0])
        dump_tensor = dump_tensor.to(device)

        return dump_tensor


def tree_unflatten_and_drop_nones(values: list, spec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, TreeSpec):
        raise ValueError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(spec)}."
        )
    if len(values) != spec.num_leaves:
        raise ValueError(
            f"tree_unflatten(values, spec): `values` has length {len(values)} "
            f"but the spec refers to a pytree that holds {spec.num_leaves} "
            f"items ({spec})."
        )
    if isinstance(spec, LeafSpec):
        return values[0]

    unflatten_fn = SUPPORTED_NODES[spec.type].unflatten_fn

    # Recursively unflatten the children
    start = 0
    end = 0
    child_pytrees = []
    context = []

    if spec.context:
        spec_context = spec.context
    else:
        spec_context = [None] * len(spec.children_specs)

    for child_context, child_spec in zip(spec_context, spec.children_specs):
        end += child_spec.num_leaves
        child_value = values[start:end]
        start = end

        child_value = [c_v for c_v in child_value if c_v is not None]

        if len(child_value) > 0:
            child_spec.num_leaves = len(child_value)
            context.append(child_context)
            child_pytrees.append(tree_unflatten_and_drop_nones(child_value, child_spec))

    return unflatten_fn(child_pytrees, context)


class PipeCommunicator:
    """
    communication class to send activations to the next pipeline stage
    - A state (meta) is kept for the tensor shapes, assumption is that the shapes do not change unless reset() is called
    - Internally, communication only happens in tuples
    - The activations are reconstructed using the meta information
    """

    def __init__(
        self,
        local_device: torch.device,
        recv_grads: bool,
        recv_data: bool = True,
        use_continuous_recommunication: bool = False,
    ):
        self.local_device = local_device
        self._recv_grads = recv_grads
        self._recv_data = recv_data

        self._use_continuous_recommunication: bool = use_continuous_recommunication

        self.communication_meta_tree_spec: Optional[TreeSpec] = None
        self.communication_meta_base_flatten: Optional[list[CommunicationMetaBase]] = None

        self.activation_recv_buffer: list[Optional[torch.Tensor]] = list()
        self.gradient_recv_buffer: list[Optional[torch.Tensor]] = list()

        self._recv_meta_data_tensor_is_same = torch.tensor(data=[False], dtype=torch.bool, device=self.local_device)
        self._recv_meta_data_tensor_len = torch.tensor(data=[0], dtype=torch.long, device=self.local_device)

    def _send_meta_data(
        self, data: PyTree, target_global_rank: int
    ) -> tuple[None, bool] | tuple[list[torch.Tensor], bool]:
        if (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ) and not self._use_continuous_recommunication:
            return None, False

        flatten_data, communication_meta_tree_spec = tree_flatten(data)

        if self.communication_meta_base_flatten is None:
            max_buffer_size: list[None] | list[int] = [None] * len(flatten_data)
        else:
            if communication_meta_tree_spec == self.communication_meta_tree_spec:
                max_buffer_size = [meta.tensor_shape[0] for meta in self.communication_meta_base_flatten]
            else:
                max_buffer_size = [None] * len(flatten_data)

        meta_and_tensor_map_list = list(
            zip(
                *map(
                    partial(map_to_meta_and_tensor_fn, self.local_device),
                    flatten_data,
                    max_buffer_size,
                )
            )
        )

        # TODO: python ZIP has the wrong output type.
        #  so this is a really hacky overwrite. Find a better way in the future.
        communication_meta_base_flatten: list[CommunicationMetaBase] = meta_and_tensor_map_list[0]  # type: ignore
        flatten_tensor: list[torch.Tensor] = meta_and_tensor_map_list[1]  # type: ignore

        if self._use_continuous_recommunication and (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ):
            is_same = communication_meta_tree_spec == self.communication_meta_tree_spec
            is_meta_same = list(
                map(
                    lambda x, y: x.compare(y),
                    communication_meta_base_flatten,
                    self.communication_meta_base_flatten,
                )
            )
            is_same = is_same and (False not in is_meta_same)

            tensor_is_same = torch.tensor(
                [is_same],
                device=self.local_device,
                dtype=torch.bool,  # type: ignore
            )

            torch.distributed.send(
                tensor_is_same,
                target_global_rank,
            )

            if is_same:
                return flatten_tensor, False

        self.communication_meta_tree_spec = communication_meta_tree_spec
        self.communication_meta_base_flatten = communication_meta_base_flatten

        assert self.communication_meta_base_flatten is not None
        communication_meta_base_tree = tree_unflatten(
            self.communication_meta_base_flatten, self.communication_meta_tree_spec
        )
        communication_meta_base_tree_dump, tensor_len = _dump_to_pickle_tensor(x=communication_meta_base_tree)

        communication_meta_base_tree_dump = communication_meta_base_tree_dump.to(self.local_device)

        tensor_len_tensor = torch.tensor(data=[tensor_len], dtype=torch.long, device=self.local_device)

        torch.distributed.send(
            tensor_len_tensor,
            target_global_rank,
        )

        torch.distributed.send(
            communication_meta_base_tree_dump,
            target_global_rank,
        )

        if self._recv_grads and self.communication_meta_base_flatten is not None:
            self.gradient_recv_buffer = [
                (
                    torch.zeros(
                        meta.tensor_shape,
                        dtype=meta.tensor_dtype,
                        device=self.local_device,  # type: ignore
                    )
                    if meta.requires_grad
                    else None
                )
                for meta in self.communication_meta_base_flatten
            ]

        return flatten_tensor, True

    def _recv_meta_data(self, origin_global_rank: int) -> bool:
        if (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ) and not self._use_continuous_recommunication:
            return False

        if self._use_continuous_recommunication and (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ):
            torch.distributed.recv(
                self._recv_meta_data_tensor_is_same,
                origin_global_rank,
            )

            if self._recv_meta_data_tensor_is_same[0]:
                return False

        self.communication_meta_tree_spec = None
        self.communication_meta_base_flatten = None

        torch.distributed.recv(
            self._recv_meta_data_tensor_len,
            origin_global_rank,
        )

        communication_meta_base_tree_dump_tensor = torch.zeros(
            size=list(
                self._recv_meta_data_tensor_len,
            ),
            dtype=torch.uint8,
            device=self.local_device,  # type: ignore
        )

        torch.distributed.recv(
            communication_meta_base_tree_dump_tensor,
            origin_global_rank,
        )

        communication_meta_base_tree_dump: bytes = bytes(communication_meta_base_tree_dump_tensor.tolist())
        communication_meta_base_tree = pickle.loads(communication_meta_base_tree_dump)

        (
            self.communication_meta_base_flatten,
            self.communication_meta_tree_spec,
        ) = tree_flatten(communication_meta_base_tree)

        # allocate buffer to receive the data later
        if self._recv_data:
            self.activation_recv_buffer = [
                torch.zeros(
                    meta.tensor_shape,
                    dtype=meta.tensor_dtype,
                    device=self.local_device,  # type: ignore
                )
                for meta in self.communication_meta_base_flatten
            ]

        return True

    def send_data(self, data: PyTree, target_global_rank: int) -> None:
        flatten_tensor, _ = self._send_meta_data(data, target_global_rank)
        flatten_data, _ = tree_flatten(data)

        if flatten_tensor is None:
            assert self.communication_meta_base_flatten is not None
            flatten_tensor = list(
                map(
                    partial(map_lower_to_tensor_fn, self.local_device),
                    flatten_data,
                    self.communication_meta_base_flatten,
                )
            )

        for tensor in flatten_tensor:
            torch.distributed.send(
                tensor,
                target_global_rank,
            )

    def recv_data(
        self,
        origin_global_rank: int,
    ) -> PyTree:
        self._recv_meta_data(origin_global_rank)

        assert self.communication_meta_base_flatten is not None
        assert self.communication_meta_tree_spec is not None

        recvd_flatten = list()
        for tensor, meta in zip(self.activation_recv_buffer, self.communication_meta_base_flatten):
            assert tensor is not None

            torch.distributed.recv(
                tensor,
                origin_global_rank,
            )
            tensor_own = tensor.clone().detach()
            if meta.requires_grad:
                tensor_own.requires_grad = True

            recvd_flatten.append(tensor_own)

        recvd_flatten = list(
            map(
                map_loads_from_pickle_tensor_fn,
                recvd_flatten,
                self.communication_meta_base_flatten,
            )
        )

        recvd = tree_unflatten(recvd_flatten, self.communication_meta_tree_spec)

        return recvd

    def send_gradients(self, data: PyTree, target_global_rank: int) -> None:
        assert (
            self.communication_meta_tree_spec is not None
        ), "communication_meta_activations is None; run send_activation_meta and recv_activation_meta first"

        assert self.communication_meta_base_flatten is not None

        flatten_tensor, _ = tree_flatten(data)

        for tensor, tensor_meta in zip(flatten_tensor, self.communication_meta_base_flatten):
            if not tensor_meta.requires_grad:
                if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and tensor.grad is not None:
                    logger.warning("Grad available for tensor that does not require grad and that is not communicated")
                continue

            assert tensor.grad is not None

            torch.distributed.send(
                tensor.grad,
                target_global_rank,
            )

    def recv_gradients(
        self,
        data: PyTree,
        origin_global_rank: int,
    ) -> GradOutput:
        assert (
            self.communication_meta_tree_spec is not None
        ), "communication_meta_activations is None; run send_activation_meta and recv_activation_meta first"

        assert self.communication_meta_base_flatten is not None

        # receive buffer is always a list
        out_tensor_flatten: list[Optional[torch.Tensor]] = list()
        grad_tensor_flatten: list[Optional[torch.Tensor]] = list()

        flatten_data, _ = tree_flatten(data)

        for grad, data, meta in zip(
            self.gradient_recv_buffer,
            flatten_data,
            self.communication_meta_base_flatten,
        ):
            if not meta.requires_grad:
                grad_tensor_flatten.append(None)
                out_tensor_flatten.append(None)
                continue

            assert isinstance(grad, torch.Tensor)

            torch.distributed.recv(
                grad,
                origin_global_rank,
            )

            grad_tensor_flatten.append(grad)
            out_tensor_flatten.append(data)

        out_tree = tree_unflatten_and_drop_nones(out_tensor_flatten, self.communication_meta_tree_spec)
        grad_tree = tree_unflatten_and_drop_nones(grad_tensor_flatten, self.communication_meta_tree_spec)

        # print("grad_tree:", grad_tree)

        return GradOutput(tensors=out_tree, grad_tensors=grad_tree)

    def reset_communication_meta(self, use_continuous_recommunication: Optional[bool] = None) -> None:
        self.communication_meta_tree_spec = None
        self.communication_meta_base_flatten = None

        self.activation_recv_buffer = list()
        self.gradient_recv_buffer = list()

        if use_continuous_recommunication is not None:
            self._use_continuous_recommunication = use_continuous_recommunication


class ModelParallelCommunicator:
    """
    communication class to send activations to the next pipeline stage
    - A state (meta) is kept for the tensor shapes, assumption is that the shapes do not change unless reset() is called
    - Internally, communication only happens in tuples
    - The activations are reconstructed using the meta information
    """

    def __init__(
        self,
        topology: Topology,
        use_continuous_recommunication: bool = False,
    ):
        self.topology = topology
        self.local_device = self.topology.device
        assert self.local_device is not None
        self._use_continuous_recommunication: bool = use_continuous_recommunication

        self.communication_meta_tree_spec: Optional[TreeSpec] = None
        self.communication_meta_base_flatten: Optional[list[CommunicationMetaBase]] = None

        self.activation_recv_buffer: list[Optional[torch.Tensor]] = list()

        self._recv_meta_data_tensor_is_same = torch.tensor(data=[False], dtype=torch.bool, device=self.local_device)
        self._recv_meta_data_tensor_len = torch.tensor(data=[0], dtype=torch.long, device=self.local_device)

    def _send_meta_data(self, data: PyTree) -> tuple[None, bool] | tuple[list[torch.Tensor], bool]:
        if (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ) and not self._use_continuous_recommunication:
            return None, False

        flatten_data, communication_meta_tree_spec = tree_flatten(data)

        if self.communication_meta_base_flatten is None:
            max_buffer_size: list[None] | list[int] = [None] * len(flatten_data)
        else:
            if communication_meta_tree_spec == self.communication_meta_tree_spec:
                max_buffer_size = [meta.tensor_shape[0] for meta in self.communication_meta_base_flatten]
            else:
                max_buffer_size = [None] * len(flatten_data)

        meta_and_tensor_map_list = list(
            zip(
                *map(
                    partial(map_to_meta_and_tensor_fn, self.local_device),
                    flatten_data,
                    max_buffer_size,
                )
            )
        )

        # TODO: python ZIP has the wrong output type. so this is a really hacky overwrite.
        #  Find a better way in the future.
        communication_meta_base_flatten: list[CommunicationMetaBase] = meta_and_tensor_map_list[0]  # type: ignore
        flatten_tensor: list[torch.Tensor] = meta_and_tensor_map_list[1]  # type: ignore

        if self._use_continuous_recommunication and (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ):
            is_same = communication_meta_tree_spec == self.communication_meta_tree_spec
            is_meta_same = list(
                map(
                    lambda x, y: x.compare(y),
                    communication_meta_base_flatten,
                    self.communication_meta_base_flatten,
                )
            )
            is_same = is_same and (False not in is_meta_same)

            tensor_is_same = torch.tensor(
                [is_same],
                device=self.local_device,
                dtype=torch.bool,  # type: ignore
            )

            torch.distributed.broadcast(
                tensor_is_same,
                torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
                group=self.topology.model_parallel_group,
            )

            if is_same:
                return flatten_tensor, False

        self.communication_meta_tree_spec = communication_meta_tree_spec
        self.communication_meta_base_flatten = communication_meta_base_flatten

        assert self.communication_meta_base_flatten is not None
        communication_meta_base_tree = tree_unflatten(
            self.communication_meta_base_flatten, self.communication_meta_tree_spec
        )
        communication_meta_base_tree_dump, tensor_len = _dump_to_pickle_tensor(x=communication_meta_base_tree)

        communication_meta_base_tree_dump = communication_meta_base_tree_dump.to(self.local_device)

        tensor_len_tensor = torch.tensor(data=[tensor_len], dtype=torch.long, device=self.local_device)

        torch.distributed.broadcast(
            tensor_len_tensor,
            torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
            group=self.topology.model_parallel_group,
        )

        torch.distributed.broadcast(
            communication_meta_base_tree_dump,
            torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
            group=self.topology.model_parallel_group,
        )

        return flatten_tensor, True

    def _recv_meta_data(
        self,
    ) -> bool:
        if (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ) and not self._use_continuous_recommunication:
            return False

        if self._use_continuous_recommunication and (
            self.communication_meta_tree_spec is not None and self.communication_meta_base_flatten is not None
        ):
            torch.distributed.broadcast(
                self._recv_meta_data_tensor_is_same,
                torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
                group=self.topology.model_parallel_group,
            )

            if self._recv_meta_data_tensor_is_same[0]:
                return False

        self.communication_meta_tree_spec = None
        self.communication_meta_base_flatten = None

        torch.distributed.broadcast(
            self._recv_meta_data_tensor_len,
            torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
            group=self.topology.model_parallel_group,
        )

        communication_meta_base_tree_dump_tensor = torch.zeros(
            size=list(
                self._recv_meta_data_tensor_len,
            ),
            dtype=torch.uint8,
            device=self.local_device,  # type: ignore
        )

        torch.distributed.broadcast(
            communication_meta_base_tree_dump_tensor,
            torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
            group=self.topology.model_parallel_group,
        )

        communication_meta_base_tree_dump: bytes = bytes(communication_meta_base_tree_dump_tensor.tolist())
        communication_meta_base_tree = pickle.loads(communication_meta_base_tree_dump)

        (
            self.communication_meta_base_flatten,
            self.communication_meta_tree_spec,
        ) = tree_flatten(communication_meta_base_tree)

        # allocate buffer to receive data later
        self.activation_recv_buffer = [
            torch.zeros(
                meta.tensor_shape,
                dtype=meta.tensor_dtype,
                device=self.local_device,  # type: ignore # type: ignore
            )
            for meta in self.communication_meta_base_flatten
        ]

        return True

    def send_data(self, data: PyTree) -> None:
        flatten_tensor, _ = self._send_meta_data(data)
        flatten_data, _ = tree_flatten(data)

        if flatten_tensor is None:
            assert self.communication_meta_base_flatten is not None
            flatten_tensor = list(
                map(
                    partial(map_lower_to_tensor_fn, self.local_device),
                    flatten_data,
                    self.communication_meta_base_flatten,
                )
            )

        for tensor in flatten_tensor:
            torch.distributed.broadcast(
                tensor,
                torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
                group=self.topology.model_parallel_group,
            )

    def recv_data(
        self,
    ) -> PyTree:
        self._recv_meta_data()

        assert self.communication_meta_base_flatten is not None
        assert self.communication_meta_tree_spec is not None

        recvd_flatten = list()
        for tensor, meta in zip(self.activation_recv_buffer, self.communication_meta_base_flatten):
            assert tensor is not None

            torch.distributed.broadcast(
                tensor,
                torch.distributed.distributed_c10d.get_global_rank(self.topology.model_parallel_group, 0),
                group=self.topology.model_parallel_group,
            )
            tensor_own = tensor.clone().detach()
            if meta.requires_grad:
                tensor_own.requires_grad = True

            recvd_flatten.append(tensor_own)

        recvd_flatten = list(
            map(
                map_loads_from_pickle_tensor_fn,
                recvd_flatten,
                self.communication_meta_base_flatten,
            )
        )

        recvd = tree_unflatten(recvd_flatten, self.communication_meta_tree_spec)

        return recvd

    def sync_data(self, data: PyTree) -> PyTree:
        if self.topology.model_parallel_rank == 0:
            self.send_data(data)
        else:
            data = self.recv_data()

        return data

    def reset_communication_meta(self, use_continuous_recommunication: Optional[bool] = None) -> None:
        self.communication_meta_tree_spec = None
        self.communication_meta_base_flatten = None

        self.activation_recv_buffer = list()

        if use_continuous_recommunication is not None:
            self._use_continuous_recommunication = use_continuous_recommunication
