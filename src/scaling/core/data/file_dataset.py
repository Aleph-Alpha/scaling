import atexit
import json
from pathlib import Path
from typing import IO, Any, Iterator, Optional

import numpy as np

from .file_handles import FileHandle, RetryableException


class FileDataset:
    """
    Dataset which loads dataset items on demand using regular file operations and seeking.

    This is usually less efficient than MemoryMapDataset, but has the advantage that all operations are retryable.
    More specifically, if accessing the underlying filesystem fails,
    reading to a memory-mapped file may kill the process with a SIGBUS error.
    With regular file operations, such failures can be caught and the operation can be retried.
    """

    def __init__(self, prefix_path: Path, load_index_to_memory: bool = False) -> None:
        # Ensure destructor is run on interpreter shutdown
        atexit.register(self.__del__)

        self.prefix_path = Path(prefix_path)
        self.load_index_to_memory = load_index_to_memory
        self.initialize()

    def initialize(self) -> None:
        # load meta data
        meta_file = FileHandle(self.file_path_meta)
        meta_dict = meta_file.retry_operation(json.load)
        meta_file.close()

        self.dtype = np.dtype(meta_dict["dtype"])

        self.index_dtype = np.dtype(meta_dict["index_dtype"])

        self.dtype_size = self.dtype.itemsize
        self.index_dtype_size = self.index_dtype.itemsize
        self.document_count = meta_dict["document_count"]

        # create file handles for bin and index files
        self._bin_file = FileHandle(self.file_path_data, "rb")
        self._index_file = FileHandle(self.file_path_index, "rb")

        self._index: Optional[np.ndarray] = None
        if self.load_index_to_memory:
            self._index = np.array(
                retry_array_from_file(
                    self._index_file,
                    dtype=self.index_dtype,
                    count=2 * len(self),
                    offset=0,
                ).reshape(len(self), 2)
            )
            self._index_file.close()

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
                    size = retry_array_from_file(
                        self._index_file,
                        dtype=self.index_dtype,
                        count=1,
                        offset=int(idx * 2 + 1) * self.index_dtype_size,
                    )
                    sizes[idx] = size

                return sizes
            else:
                size = retry_array_from_file(
                    self._index_file,
                    dtype=self.index_dtype,
                    count=1,
                    offset=int(idx * 2 + 1) * self.index_dtype_size,
                )
                return np.array(size, dtype=self.index_dtype)

    def __del__(self) -> None:
        if hasattr(self, "_bin_file"):
            del self._bin_file

        if hasattr(self, "_index_file"):
            del self._index_file

    def __getitem__(self, idx: int) -> np.ndarray:
        if not isinstance(idx, int):
            raise NotImplementedError

        assert idx < self.document_count, f"cannot retrieve document idx {idx} from {self.document_count} documents"

        if self.load_index_to_memory:
            assert self._index is not None
            start_index, size = self._index[idx].tolist()
        else:
            start_index, size = retry_array_from_file(
                self._index_file,
                dtype=self.index_dtype,
                count=2,
                offset=int(idx * 2) * self.index_dtype_size,
            )
        np_array = retry_array_from_file(
            self._bin_file,
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


def retry_array_from_file(
    file_handle: FileHandle,
    dtype: np.dtype,
    count: int,
    offset: int,
    max_attempts: int = 5,
    max_delay: int = 32,
) -> np.ndarray:
    """
    Retry reading `count` elements of `dtype` at `offset` from `file_handle`.
    """

    def f(file: IO[Any]) -> np.ndarray:
        res = np.fromfile(
            file,
            dtype=dtype,
            count=count,
            offset=offset,
        )
        # np.fromfile seeks to offset but does not reset the file pointer:
        # https://github.com/numpy/numpy/blob/4df879a/numpy/_core/src/multiarray/ctors.c#L3368
        # Since we reuse the file handle, we need to seek back to the start of the file, otherwise
        # the offset calculation within np.fromfile will be wrong.
        file.seek(0)

        # np.fromfile doesn't check errno after fread and returns an empty array if the read fails.
        # Therefore, if we get an empty array but expected to read something, we raise an exception
        # which will trigger a retry.
        if len(res) == 0 and count != 0:
            raise RetryableException(
                f"Expected to read {count} elements of dtype {dtype} "
                f"at offset {offset} from {file_handle._path}, "
                f"but got {len(res)}"
            )
        # If we read an unexpected number of elements, then something went catastrophically wrong.
        # Here we raise an exception which will not be retried.
        elif len(res) != count:
            raise Exception(
                f"Expected to read {count} elements of dtype {dtype} "
                f"at offset {offset} from {file_handle._path}, "
                f"but got {len(res)}"
            )

        return res

    return file_handle.retry_operation(f, max_attempts=max_attempts, max_delay=max_delay)
