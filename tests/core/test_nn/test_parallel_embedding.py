from typing import Optional

import pytest
import torch

from scaling.core import Topology, TopologyConfig, VocabParallelEmbedding
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.utils import dist_launcher


def run_test_parallel_embedding(
    return_dict: dict,
    model_parallel_size: Optional[int],
    vocab_size: int,
    hidden_state_size: int,
):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(  # type: ignore[call-arg]
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            model_parallel_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            pipe_parallel_size=1,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    merged_weights = torch.zeros((vocab_size, hidden_state_size), dtype=torch.float32).cuda()

    parallel_embedding = VocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_state_size,
        topology=topology,
        finetunable_token_ids=[],
    )

    # retrieve initialized weights of parallel linear layer
    if model_parallel_size == 1:
        merged_weights = parallel_embedding.weight
    else:
        merged_weights[
            topology.model_parallel_rank * (vocab_size // 2) : (topology.model_parallel_rank + 1) * (vocab_size // 2),
            :,
        ].copy_(parallel_embedding.weight)

    torch.distributed.all_reduce(merged_weights)

    embedding = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_state_size,
        device=topology.device,
    )
    # replace initialized weights of embedding layer with the one from the parallel embedding layer
    # to be able to compare the forward pass results
    embedding.weight.data.copy_(merged_weights.data)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Generate random embedding input
    x = torch.randint(low=0, high=1, size=(2,), device=topology.device)

    output_embedding = embedding(x)
    output_parallel_embedding = parallel_embedding(x)

    if model_parallel_size == 1:
        assert torch.isclose(
            output_parallel_embedding, output_embedding
        ).all(), "output from parallel implementation and default torch.nn.Embedding implementation differs"
    else:
        # in the model parallel case we encounter problems with precision so that a torch.isclose() does not cover it
        delta = (output_parallel_embedding - output_embedding).abs().mean()
        assert delta < 0.005, (
            f"output from parallel implementation and "
            f"default torch.nn.Embedding implementation differs with delta of {delta}"
        )


@pytest.mark.nn_rest
@pytest.mark.parametrize("model_parallel_size", [1, 2])
@pytest.mark.parametrize("vocab_size", [1, 8, 17, 32])
@pytest.mark.parametrize("hidden_state_size", [1, 8, 17, 32])
def test_parallel_embedding(
    model_parallel_size: Optional[int],
    vocab_size: int,
    hidden_state_size: int,
):
    """
    tests if the output from row and column parallel implementations and the default torch.nn.Linear do not differ
    """

    # Skip test if model parallel is not possible with specified in/out_feature size
    if model_parallel_size and vocab_size % model_parallel_size != 0:
        pytest.skip(
            f"cannot parallelize embedding, vocab_size ({vocab_size}) "
            f"needs to be divisible by model parallel size ({model_parallel_size})"
        )

    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_parallel_embedding,
        world_size=model_parallel_size,
        master_port=find_free_port(),
        model_parallel_size=model_parallel_size,
        vocab_size=vocab_size,
        hidden_state_size=hidden_state_size,
    )
    assert return_dict_continuously_trained_model is not None
    pass
