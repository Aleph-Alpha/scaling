from typing import Any, Dict, List, Tuple, Union

import pytest
import torch

from scaling.core import (
    LoRaConfig,
    LoRAModuleType,
    MaskedSoftmaxConfig,
    ParallelSelfAttention,
    RelativePositionEmbeddingType,
    Topology,
    TopologyConfig,
)
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from tests.core.utils import dist_launcher


def run_test_lora_forward_pass(
    return_dict: dict,
    model_parallel_size: int,
    lora_modules: List[str],
    rank: int,
    qkv_in_one: bool,
    num_kv_heads: int,
):
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            gradient_accumulation_steps=1,
            pipe_parallel_size=1,
            data_parallel_size=1,
            model_parallel_size=model_parallel_size,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    lora_config_overwrite: Dict[str, Any] = dict()
    lora_config_overwrite["parallel_modules"] = lora_modules
    lora_config_overwrite["rank"] = rank

    lora_config: LoRaConfig = LoRaConfig.from_dict(lora_config_overwrite)
    masked_softmax_config = MaskedSoftmaxConfig()

    hidden_size = 256
    batch_size = 3
    sequence_length = 16
    num_attention_heads = 4

    self_attention_module = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        lora_config=lora_config,
        qkv_in_one=qkv_in_one,
        topology=topology,
        masked_softmax_config=masked_softmax_config,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
    )

    x = torch.randn((batch_size, sequence_length, hidden_size), device=topology.device)
    cumulative_seq_lengths = torch.tensor([0, sequence_length], device=topology.device)
    position_ids = torch.stack([torch.arange(sequence_length - 1, device=topology.device) for _ in range(batch_size)])

    output = self_attention_module.forward(
        x=x, cumulative_seq_lengths=cumulative_seq_lengths, position_ids=position_ids
    )

    assert len(self_attention_module.lora_modules) == len(
        lora_modules
    ), f"Expected the length of LoRA modules to be equal to {len(lora_modules)} \
        but was {len(self_attention_module.lora_modules)}"
    assert len(output.shape) == 3, f"Expected 3-dimensional output tensor, but got {len(output.shape)} dimensions."
    assert output.shape[0] == batch_size, f"Expected batch size of {batch_size}, but got {output.shape[0]}"


def run_test_lora_merge(
    return_dict: dict,
    model_parallel_size: int,
    lora_modules: List[str],
    rank: int,
    qkv_in_one: bool,
    num_kv_heads: int,
):
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            gradient_accumulation_steps=1,
            data_parallel_size=1,
            pipe_parallel_size=1,
            model_parallel_size=model_parallel_size,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    lora_config_overwrite: Dict[str, Any] = dict()
    lora_config_overwrite["parallel_modules"] = lora_modules
    lora_config_overwrite["rank"] = rank

    lora_config = LoRaConfig.from_dict(lora_config_overwrite)
    masked_softmax_config = MaskedSoftmaxConfig()

    hidden_size = 256
    num_attention_heads = 4

    self_attention_module = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        qkv_in_one=qkv_in_one,
        lora_config=lora_config,
        topology=topology,
        masked_softmax_config=masked_softmax_config,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
    )

    # update the B (dense_out) matrix to be non-zero, so we can check if it actually changed the attention matrices
    with torch.no_grad():
        for key, module in self_attention_module.lora_modules.items():
            module.dense_out.weight = torch.nn.Parameter(
                torch.rand(
                    module.dense_out.weight.shape,
                    dtype=module.dense_out.dtype,
                    device=module.dense_out._device,
                )
            )

    if qkv_in_one:
        prev_qkv = self_attention_module.query_key_value.weight.clone()
        if LoRAModuleType.DENSE in lora_config.parallel_modules:
            dense_clone = self_attention_module.dense.weight.clone()
    else:
        prev_qkv_dict = {}
        for module_type in LoRAModuleType:
            if module_type in lora_config.parallel_modules:
                prev_qkv_dict[module_type.value] = getattr(self_attention_module, module_type.value).weight.clone()

    self_attention_module.merge_lora_weights()

    if qkv_in_one:
        assert not torch.equal(
            prev_qkv, self_attention_module.query_key_value.weight
        ), "query_key_value did not change after merge"
        if LoRAModuleType.DENSE in lora_config.parallel_modules:
            assert not torch.equal(
                dense_clone, self_attention_module.dense.weight
            ), "dense matrix did not change after merge"
    else:
        for module_type in LoRAModuleType:
            if module_type in lora_config.parallel_modules:
                assert not torch.equal(
                    prev_qkv_dict[module_type.value], self_attention_module.query.weight
                ), f"{module_type.value} matrix did not change after merge"

    assert not hasattr(
        self_attention_module, "lora_modules"
    ), "LoRA weights should be removed from module after merging but were not None"
    assert self_attention_module.lora_merged_state is True, "lora_merged_state should be in the state True"


def run_test_lora_output_equal_after_merge(
    return_dict: dict,
    model_parallel_size: int,
    lora_modules: List[str],
    rank: int,
    qkv_in_one: bool,
    num_kv_heads: int,
):
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            gradient_accumulation_steps=1,
            data_parallel_size=1,
            pipe_parallel_size=1,
            model_parallel_size=model_parallel_size,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    lora_config_overwrite: Dict[str, Any] = dict()
    lora_config_overwrite["parallel_modules"] = lora_modules
    lora_config_overwrite["rank"] = rank

    lora_config = LoRaConfig.from_dict(lora_config_overwrite)
    masked_softmax_config = MaskedSoftmaxConfig()

    hidden_size = 256
    batch_size = 3
    sequence_length = 16
    num_attention_heads = 4

    self_attention_module = ParallelSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        qkv_in_one=qkv_in_one,
        lora_config=lora_config,
        topology=topology,
        masked_softmax_config=masked_softmax_config,
        relative_position_embedding_type=RelativePositionEmbeddingType.NONE,
    )

    with torch.no_grad():
        for _, module in self_attention_module.lora_modules.items():
            module.dense_out.weight = torch.nn.Parameter(
                torch.rand(
                    module.dense_out.weight.shape,
                    dtype=module.dense_out.dtype,
                    device=module.dense_out._device,
                )
            )

    x = torch.randn((batch_size, sequence_length, hidden_size), device=topology.device)
    cumulative_seq_lengths = torch.tensor([0, sequence_length], device=topology.device)
    position_ids = torch.stack([torch.arange(sequence_length - 1, device=topology.device) for _ in range(batch_size)])

    unmerged_output = self_attention_module.forward(
        x=x, cumulative_seq_lengths=cumulative_seq_lengths, position_ids=position_ids
    )

    self_attention_module.merge_lora_weights()
    merged_output = self_attention_module.forward(
        x=x, cumulative_seq_lengths=cumulative_seq_lengths, position_ids=position_ids
    )

    assert torch.allclose(
        merged_output, unmerged_output, atol=1e6
    ), "Output of merged and unmerged of LoRA-SelfAttention module does not match."


@pytest.mark.lora
@pytest.mark.parametrize("rank", [4, 16])
@pytest.mark.parametrize("lora_modules", [["query"], ["value", "dense"], ["query", "key"]])
@pytest.mark.parametrize("model_parallelism", [1, 2])
@pytest.mark.parametrize("num_kv_heads", [None, 2])
@pytest.mark.parametrize("qkv_in_one", [True, False])
def test_lora_forward_pass(
    lora_modules: List[str],
    rank: int,
    model_parallelism: Tuple[int],
    qkv_in_one: bool,
    num_kv_heads: Union[int, None],
):
    if num_kv_heads and qkv_in_one:
        pytest.skip("GQA won't work if QKV is in one matrix")

    dist_launcher(
        run_func=run_test_lora_forward_pass,
        world_size=model_parallelism,
        master_port=find_free_port(),
        model_parallel_size=model_parallelism,
        num_kv_heads=num_kv_heads,
        qkv_in_one=qkv_in_one,
        rank=rank,
        lora_modules=lora_modules,
    )

    pass


@pytest.mark.lora
@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("lora_modules", [["query"], ["value", "dense"], ["value", "key"]])
@pytest.mark.parametrize("model_parallelism", [1, 2])
@pytest.mark.parametrize("num_kv_heads", [None, 2])
@pytest.mark.parametrize("qkv_in_one", [True, False])
def test_lora_merge(
    lora_modules: List[str],
    rank: int,
    model_parallelism: Tuple[int],
    num_kv_heads: Union[int, None],
    qkv_in_one: bool,
):
    if num_kv_heads and qkv_in_one:
        pytest.skip("GQA won't work if QKV is in one matrix")

    dist_launcher(
        run_func=run_test_lora_merge,
        world_size=model_parallelism,
        master_port=find_free_port(),
        model_parallel_size=model_parallelism,
        num_kv_heads=num_kv_heads,
        qkv_in_one=qkv_in_one,
        rank=rank,
        lora_modules=lora_modules,
    )

    pass


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("lora_modules", [["query"], ["value", "dense"], ["value", "key"]])
@pytest.mark.parametrize("model_parallelism", [1, 2])
@pytest.mark.parametrize("num_kv_heads", [None, 2])
@pytest.mark.parametrize("qkv_in_one", [True, False])
def test_lora_output_equal_after_merge(
    lora_modules: List[str],
    rank: int,
    model_parallelism: Tuple[int],
    num_kv_heads: Union[int, None],
    qkv_in_one: bool,
):
    if num_kv_heads and qkv_in_one:
        pytest.skip("GQA won't work if QKV is in one matrix")

    dist_launcher(
        run_func=run_test_lora_output_equal_after_merge,
        world_size=model_parallelism,
        master_port=find_free_port(),
        model_parallel_size=model_parallelism,
        num_kv_heads=num_kv_heads,
        qkv_in_one=qkv_in_one,
        rank=rank,
        lora_modules=lora_modules,
    )

    pass
