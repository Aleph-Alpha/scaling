import pytest

from scaling.core import pipe_partition_uniform


@pytest.mark.nn_rest
@pytest.mark.parametrize("pipe_parallel_size", [1, 2, 3, 5, 6, 17, 32])
@pytest.mark.parametrize("number_of_layers", [1, 2, 3, 5, 6, 17, 32, 64, 73, 128])
def test_pipe_partition_uniform(pipe_parallel_size: int, number_of_layers: int):
    if number_of_layers < pipe_parallel_size:
        with pytest.raises(AssertionError):
            pipe_partition_uniform(item_count=number_of_layers, partition_count=pipe_parallel_size)
        return

    partitions = pipe_partition_uniform(item_count=number_of_layers, partition_count=pipe_parallel_size)
    lengths = [coordinate.length for coordinate in partitions]

    # each pipe rank needs one partition
    assert len(partitions) == pipe_parallel_size, "got different count of partitions than pipe_parallel_size"

    # make sure that the length of partitions is uniform
    # for a uniform distribution there can only be one or two lengths
    assert 0 <= len(set(lengths)) <= 2, f"no uniform distribution, got lengths {lengths}"

    # make sure there is no empty pipeline stage
    assert min(lengths) > 0, f"one pipeline stage without layers; number of layers: {lengths}"

    # make sure maximum and minimum are at most one layer apart
    assert max(lengths) - min(lengths) <= 1, f"no uniform distribution, got lengths {lengths}"
