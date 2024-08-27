import json
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np


class MemoryMapDataset:
    """
    Dataset based on a memory map allowing for fast access to any data item

    We are using memory-mapped files because we want to access small segments of our large training files,
    without reading the entire file into memory because those files are too large to fit into memory.

    Based on numpy.memmap: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    It creates an `ndarray` backed by a memory buffer that is mapped to a file.
    """

    def __init__(self, prefix_path: Path, load_index_to_memory: bool = False) -> None:
        import atexit

        atexit.register(self.__del__)

        self.prefix_path = Path(prefix_path)
        self.load_index_to_memory = load_index_to_memory
        assert self.file_path_data.is_file(), f"cannot initialize memory map, file not found: {self.file_path_data}"
        assert self.file_path_index.is_file(), f"cannot initialize memory map, file not found: {self.file_path_index}"
        assert self.file_path_meta.is_file(), f"cannot initialize memory map, file not found: {self.file_path_meta}"
        self.initialize()

    def initialize(self) -> None:
        # load index
        meta_dict = json.loads(self.file_path_meta.read_text())

        self.dtype = np.dtype(meta_dict["dtype"])

        self.index_dtype = np.dtype(meta_dict["index_dtype"])

        self.dtype_size = self.dtype.itemsize
        self.index_dtype_size = self.index_dtype.itemsize
        self.document_count = meta_dict["document_count"]

        # open memory map
        self._bin_buffer_mmap = np.memmap(self.file_path_data, mode="r", order="C", dtype=self.dtype)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)  # type: ignore

        self._bin_buffer_index_mmap = np.memmap(self.file_path_index, mode="r", order="C", dtype=self.index_dtype)
        self._bin_buffer_index = memoryview(self._bin_buffer_index_mmap)  # type: ignore

        self._index: Optional[np.ndarray] = None
        if self.load_index_to_memory:
            self._index = np.array(
                np.frombuffer(
                    self._bin_buffer_index,
                    dtype=self.index_dtype,
                    count=2 * len(self),
                    offset=0,
                ).reshape(len(self), 2)
            )
            self._bin_buffer_index_mmap._mmap.close()  # type: ignore
            del self._bin_buffer_index_mmap

    @property
    def file_path_data(self) -> Path:
        """
        file name of the file containing data
        """
        return Path(str(self.prefix_path) + ".bin")

    @property
    def file_path_index(self) -> Path:
        """
        file name of the file containing the index to data items
        """
        return Path(str(self.prefix_path) + ".idx")

    @property
    def file_path_meta(self) -> Path:
        """
        file name of the file containing the index to data items
        """
        return Path(str(self.prefix_path) + ".meta.json")

    def sizes(self, idx: Optional[int] = None) -> np.ndarray:
        """
        token counts of the documents
        """
        if self.load_index_to_memory:
            assert self._index is not None
            if idx is None:
                return self._index[:, 1]
            else:
                return self._index[idx, 1]
        else:
            if idx is None:
                sizes = np.zeros((len(self),), dtype=self.index_dtype)
                for idx in range(len(self)):
                    size = np.frombuffer(
                        self._bin_buffer_index,
                        dtype=self.index_dtype,
                        count=1,
                        offset=int(idx * 2 + 1) * self.index_dtype_size,
                    )
                    sizes[idx] = size

                return sizes
            else:
                size = np.frombuffer(
                    self._bin_buffer_index,
                    dtype=self.index_dtype,
                    count=1,
                    offset=int(idx * 2 + 1) * self.index_dtype_size,
                )
                return np.array(size, dtype=self.index_dtype)

    def __del__(self) -> None:
        if hasattr(self, "_bin_buffer_mmap"):
            self._bin_buffer_mmap._mmap.close()  # type: ignore
            del self._bin_buffer_mmap

        if hasattr(self, "_bin_buffer_index_mmap"):
            self._bin_buffer_index_mmap._mmap.close()  # type: ignore
            del self._bin_buffer_index_mmap

    def __getitem__(self, idx: int) -> np.ndarray:
        if not isinstance(idx, int):
            raise NotImplementedError

        assert idx < self.document_count, f"cannot retrieve document idx {idx} from {self.document_count} documents"

        if self.load_index_to_memory:
            assert self._index is not None
            start_index, size = self._index[idx].tolist()
        else:
            start_index, size = np.frombuffer(
                self._bin_buffer_index,
                dtype=self.index_dtype,
                count=2,
                offset=int(idx * 2) * self.index_dtype_size,
            )
        np_array = np.frombuffer(
            self._bin_buffer,
            dtype=self.dtype,
            count=int(size),
            offset=int(start_index) * self.dtype_size,
        )
        return np_array

    def __len__(self) -> int:
        return self.document_count

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


class MemoryMapDatasetBuilder(MemoryMapDataset):
    """
    Builder for a memory map allowing writes to the dataset
    """

    def __init__(
        self,
        prefix_path: Path,
        dtype: np.dtype = np.dtype(np.int32),
        index_dtype: np.dtype = np.dtype(np.int64),
    ):
        """
        data_prefix (`Path`)
            path to a memory map

        dtype (`np.dtype`)
            data type that the data will be stored in for the memory-mapped file
            so that we can build up a pointer to every data item
        """
        self.prefix_path = prefix_path
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.initialize()

    def initialize(self) -> None:
        assert not self.file_path_data.is_file(), f"data file already exists: {self.file_path_data}"
        assert not self.file_path_index.is_file(), f"index file already exists: {self.file_path_index}"
        assert (
            self.file_path_data.parent == self.file_path_index.parent
        ), "index file and data file are not in same directory"

        # create parent directory
        self.file_path_data.parent.mkdir(exist_ok=True, parents=True)

        self.data_file = open(self.file_path_data, "wb")
        self.index_file = open(self.file_path_index, "wb")
        self.current_index = 0
        self.document_count = 0

    def add(self, np_array: np.ndarray) -> None:
        """
        adds a single one dimensional np array to the dataset
        """
        assert len(np_array.shape) == 1, "cannot add arrays of more than one dimension"
        np_array = np_array.astype(dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))

        document_length = len(np_array)
        index_array = np.array([self.current_index, document_length]).astype(self.index_dtype)
        self.index_file.write(index_array.tobytes(order="C"))
        self.current_index += document_length
        self.document_count += 1

    def finalize(self) -> None:
        """
        finalizes the creation of the dataset by closing the data file and writing the index
        """
        assert (
            self.data_file.closed is False and self.index_file.closed is False
        ), "The Builder has been finalized already"

        self.data_file.close()
        self.index_file.close()

        index_dict = {
            "dtype": np.dtype(self.dtype).name,
            "index_dtype": np.dtype(self.index_dtype).name,
            "document_count": self.document_count,
        }
        json.dump(index_dict, open(self.file_path_meta, "w"))

    def __del__(self) -> None:
        if hasattr(self, "data_file"):
            self.data_file.close()
        if hasattr(self, "index_file"):
            self.index_file.close()

    def __enter__(self) -> "MemoryMapDatasetBuilder":
        return self

    def __exit__(self, *_args: tuple, **_kwargs: dict) -> Literal[False]:
        """
        Perform cleanup actions and always propagate exceptions.

        Args:
            exc_type: The type of the exception raised in the managed block, if any.
            exc_val: The exception instance raised, if any.
            exc_tb: The traceback object associated with the exception, if any.

        Returns:
            Always returns False, indicating that any exception raised should not be suppressed.
        """
        self.finalize()
        return False
