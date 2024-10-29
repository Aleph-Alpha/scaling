import json
import shutil
import uuid
from pathlib import Path

import pytest

from scaling.core.data.pb_memory_map import PbMemoryMap, PbMemoryMapBuilder
from scaling.core.data.proto import text_image_dataset_pb2
from scaling.transformer.tokenizer import load_tokenizers


@pytest.fixture
def data():
    data = []
    fpath = Path("tests/transformer/files/dataset/text_image_data.jsonl")
    with open(fpath, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


@pytest.fixture(scope="function")
def pb_mmap_builder():
    tmp_folder = Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    tokenizer, tokenizer_no_prefix_space = load_tokenizers(Path("tests/transformer/files/alpha-001-128k.json"))

    yield PbMemoryMapBuilder(
        prefix_path=Path(tmp_folder, uuid.uuid4().hex[:8]),
        tokenizer=tokenizer,
        tokenizer_no_prefix_space=tokenizer_no_prefix_space,
        pb_datatype=text_image_dataset_pb2.TextImageExample,
    )

    shutil.rmtree(tmp_folder)


@pytest.mark.parametrize("index", [0, 1, 2, 4, 5])
def test_add(pb_mmap_builder: PbMemoryMapBuilder, data: list, index: int):
    pb_mmap_builder.add(data[index])
    pb_mmap_builder.finalize()

    with open(pb_mmap_builder.memory_map.file_path_data, "rb") as f:
        serialized = f.read()

    example = pb_mmap_builder.pb_datatype()
    example.ParseFromString(serialized)

    result = pb_mmap_builder.memory_map.pb2dict(example)
    expected = pb_mmap_builder._preprocess(data[index])
    assert set(expected.keys()) == set(result.keys())

    for k in result.keys():
        assert expected[k] == result[k]


def test_get_item(pb_mmap_builder: PbMemoryMapBuilder, data: list):
    pb_mmap = PbMemoryMap(
        prefix_path=Path("tests/transformer/files/dataset/text_image_data"),
        pb_datatype=text_image_dataset_pb2.TextImageExample,  # type: ignore
    )

    assert len(data) == len(pb_mmap)
    for index in range(len(pb_mmap)):
        pb_item = pb_mmap[index]
        expected_item = pb_mmap_builder._preprocess(data[index])
        assert set(pb_item.keys()) == set(expected_item.keys())

        for k in pb_item.keys():
            assert pb_item[k] == expected_item[k]
