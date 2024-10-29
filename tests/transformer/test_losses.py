from copy import copy
from pathlib import Path

import pytest
import torch

from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.topology.topology import Topology
from scaling.core.topology.topology_config import TopologyConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer.context.config import ContrastiveLossFunctionConfig, LossFunctionType, TransformerConfig
from scaling.transformer.context.context import TransformerContext
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.model import TransformerLayerIO
from scaling.transformer.model.losses import ContrastiveLoss, CrossEntropyLoss, create_loss_function
from tests.core.utils import dist_launcher


def construct_basic_training_config(
    cache_dir: Path,
    model_parallel_size: int,
    pipe_parallel_size: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    masked_softmax: dict = {"kernel": "flash_attention"},
):
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    data_config = {
        "data_prefixes": [Path(__file__).parents[0] / "files" / "dataset" / "dummy.jsonl"],
        "blended_dataset": {"cache_directory": cache_dir},
        "embedding_dataset": True,
        "embedding_dataset_memory_map": False,
    }

    config_dict: dict = {
        "topology": {
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "pipe_parallel_size": pipe_parallel_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_rank": 0,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.99,
            "gradient_clipping": 1.0,
            "loss_scaler": {
                "initial_scale": 16,  # set low initial loss scale to actually perform a train step in this short test
            },
            "zero": True,
        },
        "training_groups": [
            {
                "group_name": "param_group",
                "learning_rate_scheduler": {
                    "learning_rate": 0.01,
                    "learning_rate_minimum": 0.0,
                    "learning_rate_decay_style": "cosine",
                    "learning_rate_warmup_steps": 2,
                    "learning_rate_decay_iters": 10,
                },
            }
        ],
        "trainer": {
            "save_dir": str(cache_dir),
            "save_interval": 6,
            "load_dir": str(cache_dir),
            "train_iterations": 10,
            "assert_checkpoint_loaded": False,
        },
        "training": {
            "loss_function_config": {"loss_type": "contrastive_loss"},
        },
        "logger": {"log_level": "debug", "log_dir": str(cache_dir / "logs")},
        "profiler": {"profile_steps": 2, "profile_start_at_step": 1},
        "data": data_config,
        "transformer_architecture": {
            "image_encoder": True,
            "weight_tying": False,
            "vocab_size": 128000,
            "vocab_file": Path(__file__).parents[0] / "files" / "alpha-001-128k.json",
            "sequence_length": 256,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 24,
            "dropout_embedding": 0.1,
            "dropout_attention_probs": 0.1,
            "dropout_after_attention": 0.1,
            "dropout_after_mlp": 0.1,
            "lm_head": True,
            "masked_softmax": masked_softmax,
        },
    }
    return config_dict


loss_function_mapping = {"cross_entropy_loss": CrossEntropyLoss, "contrastive_loss": ContrastiveLoss}


@pytest.mark.embedding
@pytest.mark.parametrize("loss_config_name", ["cross_entropy_loss", "contrastive_loss"])
def test_basic_loss_loading(tmp_path: Path, loss_config_name: str):
    config_dict = construct_basic_training_config(
        tmp_path,
        model_parallel_size=1,
        pipe_parallel_size=1,
        world_size=1,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )

    config_dict["training"]["loss_function_config"] = {"loss_type": loss_config_name}
    config = TransformerConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = TransformerContext(config=config, topology=topology)

    loss_function = create_loss_function(context, config.training.loss_function_config)

    assert type(loss_function) is loss_function_mapping[loss_config_name], f"""Wrong loss function
        was initialized,{loss_function_mapping[loss_config_name]} was expected but {type(loss_function)} was found"""


def run_test_distributed_contrastive_loss(
    return_dict: dict,
    model_parallel_size: int,
    data_parallel_size: int,
    pipe_parallel_size: int,
    number_of_hard_negatives: int,
    activations: torch.Tensor,
):
    launch_config = LaunchConfig.from_launcher_args()
    world_size = model_parallel_size * data_parallel_size * pipe_parallel_size
    topology = Topology(
        config=TopologyConfig.from_dict(
            {
                "global_rank": launch_config.global_rank,
                "world_size": world_size,
                "local_slot": launch_config.local_slot,
                "gradient_accumulation_steps": 1,
                "pipe_parallel_size": pipe_parallel_size,
                "data_parallel_size": data_parallel_size,
                "model_parallel_size": model_parallel_size,
                "micro_batch_size": 1,
            }
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    activations = activations.to(topology.device)

    loss_config = ContrastiveLossFunctionConfig(
        loss_type=LossFunctionType.CONTRASTIVE_LOSS,
        use_instructions=False,
        query_side_only=False,
        scale=1,
        log_verbose_metrics=True,
        number_of_hard_negatives=number_of_hard_negatives,
    )
    loss_fn = ContrastiveLoss(topology, loss_config)

    position_ids = torch.tensor([])
    cumulative_seq_lengths_padded = torch.tensor([])

    layer_io = TransformerLayerIO(
        activations=activations, position_ids=position_ids, cumulative_seq_lengths_padded=cumulative_seq_lengths_padded
    )
    dummy_batch = TextDatasetBatch()
    loss, metrics = loss_fn(layer_io, dummy_batch)
    extracted_metrics = [val.item() for val in metrics.values()]

    return_dict[f"loss_global_rank_{topology.device}"] = loss.item()
    return_dict[f"metrics_global_rank_{topology.device}"] = extracted_metrics


@pytest.mark.embedding
@pytest.mark.parametrize("number_of_hard_negatives", [0, 1])
def test_distributed_contrastive_loss_across_parallelisms(number_of_hard_negatives: int):
    metrics = []
    losses = []
    hidden_dim = 32

    base_activations = torch.randn(10 * (number_of_hard_negatives + 2), hidden_dim)
    for configuration in [[1, 1, 1], [2, 1, 1], [1, 1, 2], [1, 2, 1]]:
        model_parallel_size, data_parallel_size, pipe_parallel_size = configuration
        if data_parallel_size == 1:
            activations = copy(base_activations).repeat((2, 1))
        else:
            activations = copy(base_activations)

        ws = model_parallel_size * data_parallel_size * pipe_parallel_size

        return_dict_contrastive_loss_results = dist_launcher(
            run_func=run_test_distributed_contrastive_loss,
            world_size=ws,
            master_port=find_free_port(),
            model_parallel_size=model_parallel_size,
            pipe_parallel_size=pipe_parallel_size,
            data_parallel_size=data_parallel_size,
            number_of_hard_negatives=number_of_hard_negatives,
            activations=activations,
        )
        return_dict_contrastive_loss_results
        losses.extend([val for key, val in return_dict_contrastive_loss_results.items() if key.startswith("loss")])
        metrics.extend([val for key, val in return_dict_contrastive_loss_results.items() if key.startswith("metrics")])

    assert torch.all(
        torch.isclose(torch.tensor(losses), torch.tensor(losses[0]))
    ), "Losses did not equal across different parallelism schemes"
    assert torch.all(
        torch.tensor(
            [
                torch.all(torch.isclose(torch.tensor(metrics)[:, i], torch.tensor(metrics)[:, i][0]))
                for i in enumerate(metrics[0])
            ]
        )
    ), "Metrics did not equal across different parallelism schemes"
