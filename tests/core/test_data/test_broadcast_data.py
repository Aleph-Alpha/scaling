import numpy as np
import pytest
import torch

from scaling.core.data.broadcast_data import _MAX_DATA_DIM, _build_tensor_sizes, _unpack_sizes


@pytest.mark.short
def test_build_tensor_sizes_not_on_rank_0():
    no_tensors = 3
    tensors = [torch.randn(20)] * no_tensors
    model_parallel_rank = 4
    result = _build_tensor_sizes(tensors, model_parallel_rank)
    assert all([x == -1 for x in result])
    assert len(result) == no_tensors * _MAX_DATA_DIM


@pytest.mark.short
def test_build_tensor_sizes_on_rank_0_bigger_than_max_data():
    dimensions = [1 for _ in range(1, _MAX_DATA_DIM + 2)]
    tensors = [torch.randn(dimensions)]
    model_parallel_rank = 0
    with pytest.raises(AssertionError):
        _build_tensor_sizes(tensors, model_parallel_rank)


@pytest.mark.short
def test_build_tensor_sizes():
    tensors = [torch.randn(1, 2), torch.randn(4, 7, 8)]
    result = _build_tensor_sizes(tensors, 0)
    assert result == [1, 2, -1, -1, -1, -1, -1, -1, 4, 7, 8, -1, -1, -1, -1, -1]


@pytest.mark.short
@pytest.mark.parametrize("number_of_tensors", [1, 40, 2])
@pytest.mark.parametrize("dimensions", [[i for i in range(1, 5)], [1], [4, 3, 5]])
def test_build_tensor_sizes_multiple(number_of_tensors, dimensions):
    tensors = [torch.randn(dimensions)] * number_of_tensors
    model_parallel_rank = 0
    result = _build_tensor_sizes(tensors, model_parallel_rank)
    assert result == (dimensions + [-1] * (_MAX_DATA_DIM - len(dimensions))) * number_of_tensors


@pytest.mark.short
def test_unpack_sizes():
    input = [1, 2, -1, -1, -1, -1, -1, -1, 4, 7, 8, -1, -1, -1, -1, -1]
    sizes, number_of_elements = _unpack_sizes(input)
    assert sizes == [[1, 2], [4, 7, 8]]
    assert number_of_elements == [1 * 2, 4 * 7 * 8]


@pytest.mark.short
def test_unpack_sizes_full():
    expected_first = [1, 2, 3, 4, 5, 6, 7, 8]
    expected_second = [9, 10, 11, 12, 13, 14, 15, 16]
    sizes, number_of_elements = _unpack_sizes(expected_first + expected_second)
    assert sizes == [expected_first, expected_second]
    assert number_of_elements == [np.prod(expected_first), np.prod(expected_second)]
