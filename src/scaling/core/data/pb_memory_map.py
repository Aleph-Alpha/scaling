import atexit
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Type

import numpy as np
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from pydantic import BaseModel

from scaling.transformer.tokenizer import Tokenizer


class PromptImageLocation(BaseModel):
    start_index: int = 0
    end_index: int = 0


class TextImageExample(BaseModel):
    input_token_list: list[int] = []
    target_token_list: list[int] = []
    loss_mask_list: list[bool] = []
    prompt_image_data: list[bytes] = []
    prompt_image_locations: list[PromptImageLocation] = []


class Index(ABC):
    def __init__(self, file_path: Path, dtype: np.dtype):
        assert file_path.is_file(), f"cannot initialize index, file not found: {file_path}"

        self.dtype = dtype
        self._bin_buffer_index = np.memmap(file_path, mode="r", order="C", dtype=self.dtype)

    @abstractmethod
    def get(self, index: int) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def size(self, indices: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError


class InMemoryIndex(Index):
    def __init__(self, file_path: Path, dtype: np.dtype, length: int):
        super().__init__(file_path=file_path, dtype=dtype)
        self.length = length
        self._index = np.array(
            np.frombuffer(
                self._bin_buffer_index,
                dtype=self.dtype,
                count=2 * length,
                offset=0,
            ).reshape(length, 2)
        )

        del self._bin_buffer_index

    def get(self, index: int) -> tuple[int, int]:
        start_index, size = self._index[index].tolist()
        return int(start_index), int(size)

    def size(self, indices: Iterable[int]) -> np.ndarray:
        raise NotImplementedError


class MemoryMapIndex(Index):
    def get(self, index: int) -> tuple[int, int]:
        offset = int(index * 2) * self.dtype.itemsize
        start_index, size = np.frombuffer(
            self._bin_buffer_index,
            dtype=self.dtype,
            offset=offset,
            count=2,
        )
        return int(start_index), int(size)

    def size(self, indices: Iterable[int]) -> Generator[int, None, None]:
        for index in indices:
            offset = int(index * 2 + 1) * self.dtype.itemsize
            size = np.frombuffer(
                self._bin_buffer_index,
                dtype=self.dtype,
                offset=offset,
                count=1,
            )

            yield int(size[0])


class PbMemoryMap:
    """
    Dataset based on a memory map allowing for fast access to any data item.

    We use memory-mapped files to access small segments of large training files
    without reading the entire file into memory, as these files are often too large
    to fit into memory.

    This class is based on numpy.memmap, which creates an `ndarray` backed by a
    memory buffer mapped to a file.

    :param prefix_path:             Path prefix for the memory map files.
    :type prefix_path:              Path
    :param pb_datatype:             Protocol Buffer message type for the data.
    :type pb_datatype:              Type[Message]
    :param initialize:              Whether to initialize the memory map upon instantiation,
                                    defaults to True.
    :type initialize:               bool, optional
    :param load_index_to_memory:    Whether to load the entire index into memory for faster
                                    access, defaults to False.
    :type load_index_to_memory:     bool, optional
    """

    def __init__(
        self,
        prefix_path: Path,
        pb_datatype: Type[Message],
        initialize: bool = True,
        load_index_to_memory: bool = False,
    ):
        atexit.register(self.__del__)
        self.prefix_path = Path(prefix_path)
        self.pb_datatype = pb_datatype

        self.load_index_to_memory = load_index_to_memory
        self.initialized = False

        self.file_path_data = Path(str(self.prefix_path) + ".bin")
        self.file_path_index = Path(str(self.prefix_path) + ".idx")
        self.file_path_meta = Path(str(self.prefix_path) + ".meta.json")

        # Variables that need to be initialized.
        self._index_dtype: np.dtype[Any] | None = None
        self._document_count: int = 0
        self._bin_buffer: Optional[np.memmap] = None
        self._index: Optional[Index] = None

        if initialize:
            self.initialize()

    @property
    def index_dtype(self) -> np.dtype[Any] | None:
        if not self.initialized:
            raise ValueError("Memory map uninitialized. Initialize before accessing " "'index_dtype' property.")
        return self._index_dtype

    @property
    def document_count(self) -> int:
        if not self.initialized:
            raise ValueError("Memory map uninitialized. Initialize before accessing " "'document_count' property.")
        return self._document_count

    def initialize(self) -> None:
        """
        Initialize the memory map.

        Raises:
            ValueError: If the memory map files do not exist.

        Note:
            This method sets the `initialized` flag to True upon successful completion.
        """

        self._assert_mmap_exists(
            file_path_data=self.file_path_data,
            file_path_index=self.file_path_index,
            file_path_meta=self.file_path_meta,
        )

        # Initialize metadata.
        meta_dict = json.loads(self.file_path_meta.read_text())
        self._index_dtype = np.dtype(meta_dict["index_dtype"])
        self._document_count = meta_dict["document_count"]

        # Open memory map.
        self._bin_buffer = np.memmap(self.file_path_data, mode="r", order="C", dtype=np.uint8)

        self._index = self._initialize_index(
            file_path=self.file_path_index,
            load_index_to_memory=self.load_index_to_memory,
            dtype=self._index_dtype,
        )

        self.initialized = True

    def size(self, indices: Iterable[int]) -> Iterable[int]:
        return self._index.size(indices=indices)  # type: ignore

    def __del__(self) -> None:
        if not self.initialized:
            return

        if hasattr(self, "_bin_buffer"):
            del self._bin_buffer

        if hasattr(self, "_bin_buffer_index"):
            del self._index

    def __getitem__(self, index: int) -> dict:
        assert index < self.document_count, f"cannot retrieve document idx {index} from {self.document_count} documents"

        start_index, size = self._index.get(index=index)  # type: ignore

        if self._bin_buffer is None:
            raise ValueError("Memory map uninitialized.")

        serialized = np.frombuffer(
            self._bin_buffer,
            dtype=np.uint8,
            offset=start_index,
            count=size,
        ).tobytes()

        example = self.pb_datatype()
        example.ParseFromString(serialized)
        item = self.pb2dict(example, TextImageExample)
        return item

    def __len__(self) -> int:
        return self.document_count

    def __iter__(self) -> Generator[dict, None, None]:
        for i in range(len(self)):
            yield self[i]

    def _assert_mmap_exists(
        self,
        file_path_data: Path,
        file_path_index: Path,
        file_path_meta: Path,
    ) -> None:
        if not file_path_data.is_file():
            raise ValueError(f"cannot initialize memory map, file not found: {file_path_data}")

        if not file_path_index.is_file():
            raise ValueError(f"cannot initialize memory map, file not found: {file_path_index}")

        if not file_path_meta.is_file():
            raise ValueError(f"cannot initialize memory map, file not found: {file_path_meta}")

    def _initialize_index(self, file_path: Path, dtype: np.dtype, load_index_to_memory: bool) -> Index:
        index: Index
        if load_index_to_memory:
            index = InMemoryIndex(file_path=file_path, dtype=dtype, length=self.document_count)
        else:  # Open memory map.
            index = MemoryMapIndex(file_path=file_path, dtype=dtype)
        return index

    def dict2pb(self, pb_message: Message, data: dict) -> Message:
        """
        Convert a dictionary to a Protocol Buffer message.

        This method recursively converts a dictionary to a Protocol Buffer message.
        It handles both repeated fields and nested messages.

        :param pb_message:  The Protocol Buffer message to be populated.
        :type pb_message:   google.protobuf.message.Message
        :param data:        The dictionary containing the data to be set in the Protocol Buffer message.
        :type data:         dict
        :return:            The populated Protocol Buffer message.
        :rtype:             google.protobuf.message.Message
        """

        for k, v in data.items():
            assert hasattr(pb_message, k)
            field = getattr(pb_message, k)
            field_descriptor = pb_message.DESCRIPTOR.fields_by_name[k]
            if field_descriptor.label == FieldDescriptor.LABEL_REPEATED:
                self._set_repeated_pb_field(field, field_descriptor, v)
            elif field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
                self.dict2pb(field, v)
            else:  # Set attribute.
                setattr(pb_message, k, v)

        return pb_message

    def pb2dict(
        self,
        pb_message: Message,
        data_model: Type[BaseModel] = TextImageExample,
    ) -> dict:
        """
        Convert a Protocol Buffer message to a dictionary.

        This method recursively converts a Protocol Buffer message to a dictionary.
        It handles both repeated fields and nested messages.

        :param pb_message:  The Protocol Buffer message to be converted.
        :type pb_message:   google.protobuf.message.Message
        :param data_model:  The Pydantic data model to use for the dictionary, defaults to TextImageExample.
        :type data_model:   Type[BaseModel], optional
        :return:            The dictionary representation of the Protocol Buffer message.
        :rtype:             dict
        """

        result = data_model().model_dump()
        for field_descriptor, value in pb_message.ListFields():
            if field_descriptor.label == FieldDescriptor.LABEL_REPEATED:
                result[field_descriptor.name] = self._get_repeated_pb_field(value, field_descriptor)
            elif field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
                result[field_descriptor.name] = self.pb2dict(value)
            else:
                result[field_descriptor.name] = value

        return result

    def _set_repeated_pb_field(self, pb_field: Any, descriptor: FieldDescriptor, data: Any) -> Any:
        if descriptor.type == FieldDescriptor.TYPE_MESSAGE:
            for item in data:
                sub_message = pb_field.add()
                self.dict2pb(sub_message, item)
        else:
            pb_field.extend(data)

        return pb_field

    def _get_repeated_pb_field(
        self,
        pb_field: Any,
        descriptor: FieldDescriptor,
    ) -> list:
        if descriptor.type == FieldDescriptor.TYPE_MESSAGE:
            return [self.pb2dict(item, PromptImageLocation) for item in pb_field]
        else:
            return list(pb_field)


class PbMemoryMapBuilder:
    """
    A builder class for creating Protocol Buffer memory maps.

    This class facilitates the construction of memory-mapped Protocol Buffer data structures.
    It handles the initialization of data and index files, preprocessing of input data,
    and sequential addition of Protocol Buffer messages to the memory map.

    :param prefix_path:                  The base path for the memory map files.
    :type prefix_path:                   Path
    :param pb_datatype:                  The Protocol Buffer message type to be used.
    :type pb_datatype:                   Type[Message]
    :param tokenizer:                    The tokenizer for processing text with prefix space.
    :type tokenizer:                     Tokenizer
    :param tokenizer_no_prefix_space:    The tokenizer for processing text without prefix space.
    :type tokenizer_no_prefix_space:     Tokenizer
    :param image_encoder_token_counts:   The number of tokens used for image encoding.
    :type image_encoder_token_counts:    int
    :param index_dtype:                  The data type for index values.
    :type index_dtype:                   type[np.signedinteger[Any]]
    """

    def __init__(
        self,
        prefix_path: Path,
        pb_datatype: Type[Message],
        tokenizer: Tokenizer,
        tokenizer_no_prefix_space: Tokenizer,
        image_encoder_token_counts: int = 144,
        index_dtype: type[np.signedinteger[Any]] = np.int64,
    ):
        self.prefix_path = prefix_path
        self.pb_datatype = pb_datatype

        self.tokenizer = tokenizer
        self.tokenizer_no_prefix_space = tokenizer_no_prefix_space
        self.image_encoder_token_counts = image_encoder_token_counts
        self.index_dtype = index_dtype

        self.memory_map = PbMemoryMap(prefix_path=prefix_path, pb_datatype=pb_datatype, initialize=False)

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the memory map.

        Raises:
            ValueError: If the memory map files already exist.
        """

        if self.memory_map.file_path_data.is_file():
            raise ValueError(f"data file already exists: {self.memory_map.file_path_data}")

        if self.memory_map.file_path_index.is_file():
            raise ValueError(f"index file already exists: {self.memory_map.file_path_index}")

        if self.memory_map.file_path_data.parent != self.memory_map.file_path_index.parent:
            raise ValueError("index file and data file are not in same directory")

        # create parent directory
        self.memory_map.file_path_data.parent.mkdir(exist_ok=True, parents=True)
        self.data_file = open(self.memory_map.file_path_data, "wb")
        self.index_file = open(self.memory_map.file_path_index, "wb")
        self.current_index = 0
        self.document_count = 0

    def add(self, data: list) -> None:
        """
        Add a list of data to the memory map.

        :param data:    The list of data to be added.
        :type data: list
        """

        pb_data = self.pb_datatype()
        data_preprocessed = self._preprocess(data=data)
        self.memory_map.dict2pb(pb_data, data_preprocessed)
        serialized = pb_data.SerializeToString()
        self.data_file.write(serialized)

        index_data = [self.current_index, len(serialized)]
        index_array = np.array(index_data).astype(self.index_dtype)
        self.index_file.write(index_array.tobytes(order="C"))

        self.current_index += len(serialized)
        self.document_count += 1

    def finalize(self) -> None:
        """
        Finalize the memory map. This method closes the data and index files
        and writes the metadata to the meta file.
        """

        self.data_file.close()
        self.index_file.close()

        index_dict = {
            "index_dtype": np.dtype(self.index_dtype).name,
            "document_count": self.document_count,
        }

        json.dump(index_dict, open(self.memory_map.file_path_meta, "w"))

    def __del__(self) -> None:
        if hasattr(self, "data_file"):
            self.data_file.close()
        if hasattr(self, "index_file"):
            self.index_file.close()

    def _preprocess(self, data: list) -> dict[str, Any]:
        token_list: list[int] = []
        loss_mask_list: list[bool] = []
        prompt_image_data: list[bytes] = []
        prompt_image_locations: list[dict] = []

        tokenizer = self.tokenizer_no_prefix_space
        image_token_id = self.tokenizer.padding_token_id

        for item in data:
            token: list[int]
            if item["type"] == "text":
                token = tokenizer.encode(item["content"])
                # Change tokenizer after first text.
                tokenizer = self.tokenizer
            elif item["type"] == "image":
                token = [image_token_id] * self.image_encoder_token_counts
                prompt_image_locations.append(
                    {
                        "start_index": len(token_list),
                        "end_index": len(token_list) + len(token),
                    }
                )
                with open(item["content"], "rb") as f:
                    image_bytes = f.read()
                prompt_image_data.append(image_bytes)
            else:
                raise NotImplementedError(f"Content type {item['type']} is not supported")

            token_list.extend(token)
            has_loss = False if item["type"] == "image" else item.get("has_loss", False)
            loss_mask_list.extend([has_loss] * len(token))

        return {
            "input_token_list": token_list[:-1],
            "target_token_list": token_list[1:],
            "loss_mask_list": loss_mask_list[1:],
            "prompt_image_data": prompt_image_data,
            "prompt_image_locations": prompt_image_locations,
        }
