# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Facebook, Inc. and its affiliates
# Copyright (c) Facebook, Inc. and its affiliates.
# SPDX-License-Identifier: MIT

import os
import struct
from functools import lru_cache
from io import BufferedReader, BufferedWriter
from itertools import accumulate
from typing import Any, Union

import numpy as np
import torch


def __best_fitting_dtype(vocab_size: int | None = None) -> type[np.integer]:
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def infer_dataset_impl(path: str) -> str | None:
    if IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == Index._HDR_MAGIC[:8]:
                return "mmap"
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")  # todo: why are we printing? Should we rather log instead?
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(
    out_file: str, impl: str, vocab_size: int | None = None
) -> Union["MMapIndexedDatasetBuilder", "IndexedDatasetBuilder"]:
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))  # type: ignore[arg-type]
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path: str, impl: str, skip_warmup: bool = False) -> torch.utils.data.Dataset | None:
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == "infer":
        impl = infer_dataset_impl(path)  # type: ignore[assignment]
    if impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == "cached" and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def read_longs(f: BufferedReader, n: int) -> np.ndarray:
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)  # type: ignore[arg-type]
    return a


def write_longs(f: BufferedWriter, a: list[int]) -> None:
    f.write(np.array(a, dtype=np.int64))  # type: ignore[arg-type]


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
}


def code(dtype: np.dtype) -> int:
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path: str) -> str:
    return prefix_path + ".idx"


def data_file_path(prefix_path: str) -> str:
    return prefix_path + ".bin"


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path: str) -> None:
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.doc_count = struct.unpack("<Q", f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)  # type: ignore[arg-type]
            self.data_offsets = read_longs(f, self._len + 1)  # type: ignore[arg-type]
            self.sizes = read_longs(f, self.s)  # type: ignore[arg-type]
            self.doc_idx = read_longs(f, self.doc_count)  # type: ignore[arg-type]

    def read_data(self, path: str) -> None:
        self.data_file = open(data_file_path(path), "rb", buffering=0)  # type: ignore[assignment]

    def check_index(self, i: int) -> None:
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self) -> None:
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, idx: int | slice) -> np.ndarray | list[np.ndarray]:
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)  # type: ignore[attr-defined]
            self.data_file.readinto(a)  # type: ignore[attr-defined]
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)  # type: ignore[attr-defined]
            self.data_file.readinto(a)  # type: ignore[attr-defined]
            offsets = list(accumulate(sizes))
            return np.split(a, offsets[:-1])

    def __len__(self) -> int:
        return self._len

    def size(self, index: int) -> np.ndarray:
        return self.sizes[index]

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class IndexedCachedDataset(IndexedDataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.cache: np.ndarray | None = None
        self.cache_index: dict[int, int] = {}

    def __getitem__(self, idx: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx : ptx + a.size])  # type: ignore[index]
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents  # type: ignore[return-value]


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.double: 8,
    }

    def __init__(self, out_file: str, dtype: np.dtype = np.int32) -> None:  # type: ignore[assignment]
        self.out_file = open(out_file, "wb")
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes: list[int] = []
        self.element_size = self.element_sizes[self.dtype]  # type: ignore[index]
        self.doc_idx = [0]

    def add_item(self, tensor: torch.Tensor) -> None:
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))  # type: ignore[arg-type]
        self.data_offsets.append(int(self.data_offsets[-1] + bytes / self.element_size))
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def finalize(self, index_file: str) -> None:
        self.out_file.close()
        index = open(index_file, "wb")
        index.write(b"TNTIDX\x00\x00")
        index.write(struct.pack("<Q", 1))
        index.write(struct.pack("<QQ", code(self.dtype), self.element_size))
        index.write(struct.pack("<QQ", len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack("<Q", len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path: str) -> None:
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class Index:
    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    @classmethod
    def writer(cls, path: str, dtype: np.dtype) -> Any:
        class _Writer(object):
            def __enter__(self) -> "_Writer":
                self._file = open(path, "wb")

                self._file.write(cls._HDR_MAGIC)
                self._file.write(struct.pack("<Q", 1))
                self._file.write(struct.pack("<B", code(dtype)))

                return self

            @staticmethod
            def _get_pointers(sizes: list[int]) -> list[int]:
                dtype_size = dtype().itemsize  # type: ignore[operator]
                address = 0
                pointers = []

                for size in sizes:
                    pointers.append(address)
                    address += size * dtype_size

                return pointers

            def write(self, sizes: list[int], doc_idx: list[int]) -> None:
                pointers = self._get_pointers(sizes)

                self._file.write(struct.pack("<Q", len(sizes)))
                self._file.write(struct.pack("<Q", len(doc_idx)))

                sizes_as_array = np.array(sizes, dtype=np.int32)
                self._file.write(sizes_as_array.tobytes(order="C"))
                del sizes_as_array
                del sizes

                pointers_as_array = np.array(pointers, dtype=np.int64)
                self._file.write(pointers_as_array.tobytes(order="C"))
                del pointers_as_array
                del pointers

                doc_idx_as_array = np.array(doc_idx, dtype=np.int64)
                self._file.write(doc_idx_as_array.tobytes(order="C"))

            def __exit__(self, *_arg: tuple, **_kwargs: dict) -> None:
                self._file.close()

        return _Writer()

    def __init__(self, path: str, skip_warmup: bool = False) -> None:
        with open(path, "rb") as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, (
                "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
            )
            version = struct.unpack("<Q", stream.read(8))
            assert (1,) == version

            (dtype_code,) = struct.unpack("<B", stream.read(1))
            self._dtype = dtypes[dtype_code]

            self._len = struct.unpack("<Q", stream.read(8))[0]
            self._doc_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        if not skip_warmup:
            # print("    warming up index mmap file...", flush=True)
            _warmup_mmap_file(path)

        self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)  # type: ignore[arg-type]
        # print("    reading sizes...", flush=True)
        self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
        # print("    reading pointers...", flush=True)
        self._pointers = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._len,
            offset=offset + self._sizes.nbytes,
        )
        # print("    reading document index...", flush=True)
        self._doc_idx = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._doc_count,
            offset=offset + self._sizes.nbytes + self._pointers.nbytes,
        )

    def __del__(self) -> None:
        self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        del self._bin_buffer_mmap

    @property
    def dtype(self) -> np.dtype:
        return self._dtype  # type: ignore[return-value]

    @property
    def sizes(self) -> np.ndarray:
        return self._sizes

    @property
    def doc_idx(self) -> np.ndarray:
        return self._doc_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self._pointers[i], self._sizes[i]

    def __len__(self) -> int:
        return self._len


class MMapIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, skip_warmup: bool = False) -> None:
        super().__init__()

        self._path: str | None = None
        self._index: Index | None = None
        self._bin_buffer: memoryview | None = None

        self._do_init(path, skip_warmup)

    def __getstate__(self) -> str | None:
        return self._path

    def __setstate__(self, state: str) -> None:
        self._do_init(state, True)

    def _do_init(self, path: str, skip_warmup: bool) -> None:
        self._path = path
        self._index = Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            # print("    warming up data mmap file...", flush=True)
            _warmup_mmap_file(data_file_path(self._path))
        # print("    creating numpy buffer of mmap...", flush=True)
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        # print("    creating memory view of numpy buffer...", flush=True)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)  # type: ignore[arg-type]

    def __del__(self) -> None:
        try:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        except:  # noqa: E722
            raise ValueError(self._path)
        del self._bin_buffer_mmap
        del self._index

    def __len__(self) -> int:
        if self._index is None:
            raise ValueError("Index cannot be None.")

        return len(self._index)

    def __getitem__(self, idx: int | slice) -> np.ndarray | list[np.ndarray]:
        if self._index is None:
            raise ValueError("Index cannot be None.")

        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)  # type: ignore[arg-type]
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)  # type: ignore[arg-type]
            return np.split(np_array, offsets[:-1])

        raise ValueError(f"idx needs to be of type int or slice, but is {type(idx)}.")

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        if self._index is None:
            raise ValueError("Index cannot be None.")

        ptr, size = self._index[idx]
        if length is None:
            length = size - offset  # type: ignore[assignment]
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)  # type: ignore[arg-type]
        return np_array

    @property
    def sizes(self) -> np.ndarray:
        if self._index is None:
            raise ValueError("Index cannot be None.")

        return self._index.sizes

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file: str, dtype: np.dtype = np.int64) -> None:  # type: ignore[assignment]
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes: list[int] = []
        self._doc_idx = [0]

    def add_item(self, tensor: torch.Tensor) -> None:
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def finalize(self, index_file: str) -> None:
        self._data_file.close()

        with Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


def get_indexed_dataset_(data_prefix: str, data_impl: str, skip_warmup: bool) -> torch.utils.data.Dataset | None:
    """Build indexed dataset."""
    indexed_dataset = make_dataset(data_prefix, data_impl, skip_warmup)
    return indexed_dataset
