from pathlib import Path

import pytest
import torch

from scaling.core import Topology
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port
from scaling.transformer import (
    TransformerConfig,
    TransformerContext,
    init_model,
)
from scaling.transformer.data.text_dataset import (
    TextDatasetBatch,
)
from scaling.transformer.data.utils import get_cumulative_seq_lengths, get_position_ids
from tests.core.utils import dist_launcher


def legacy_state_dict_to_new_state_dict(state_dict):
    """
    translate state dict keys of "old" codebase to the format used in new codebase
    """
    state_dict_ = dict()
    for k, v in state_dict.items():
        if k.endswith(".inv_freq"):
            continue
        if k == "transformer.embeddings.word_embeddings.weight":
            k_ = "_layers.0.embedding.weight"
            # add additional LMHead tied embedding
            state_dict_["_layers.3.embedding.weight"] = v
        elif k.startswith("transformer.layer0"):
            k_ = k.replace("transformer.layer0", "_layers.1")
            k_ = k_.replace(".attention.", ".self_attention.")
            k_ = k_.replace("dense_h_to_4h", "dense_in")
            k_ = k_.replace("dense_4h_to_h", "dense_out")
        elif k.startswith("transformer.norm"):
            k_ = k.replace("transformer", "_layers.2")
        else:
            raise NotImplementedError
        state_dict_[k_] = v
    return state_dict_


def run_forward_pass(
    return_dict: dict,
    config_dict: dict,
    backward_checkpoint_dir: Path,
    input: torch.Tensor,
):
    """
    Main function of the class. Runs training.
    Optionally returns list of losses.
    """
    # get configuration from launcher
    launch_config = LaunchConfig.from_launcher_args()
    config_dict["topology"]["world_size"] = launch_config.world_size
    config_dict["topology"]["global_rank"] = launch_config.global_rank
    config_dict["topology"]["local_slot"] = launch_config.local_slot

    # initialize
    config = TransformerConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    context = TransformerContext(config=config, topology=topology)
    context.initialize(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        seed=config.trainer.seed,
    )

    # initialize model, optimizer and data loader
    model = init_model(context=context)

    # Overwrite state dict of model with backward compatability checkpoint
    state_dict = legacy_state_dict_to_new_state_dict(
        torch.load(str(Path(backward_checkpoint_dir) / "state_dict.pt")),
    )
    # state_dict = torch.load(str(backward_checkpoint_dir / "state_dict.pt"))
    model.load_state_dict(state_dict)

    # convert legacy input into new transformer input
    input_token_ids = input.cuda()
    cumulative_seq_lengths = get_cumulative_seq_lengths(input_token_ids, reset_attention_mask=False)
    position_ids = get_position_ids(input_token_ids, reset_position_ids=False)
    text_dataset_batch = TextDatasetBatch(
        input_token_ids=input_token_ids,
        cumulative_seq_lengths=cumulative_seq_lengths,
        position_ids=position_ids,
    )

    # Run one forward pass with intermediate layer steps
    # Embedding layer
    embedding_output = model._layers[0](text_dataset_batch)

    # Transformer layer and subcomponents
    transformer_layer_output = model._layers[1](embedding_output)
    transformer_layer_input_layernorm_output = model._layers[1].input_layernorm(embedding_output.activations)
    transformer_layer_attention_block_output = model._layers[1].attention_block(
        embedding_output.activations,
        cumulative_seq_lengths=text_dataset_batch.cumulative_seq_lengths,
        position_ids=text_dataset_batch.position_ids,
    )
    transformer_layer_post_attention_layernorm_output = model._layers[1].post_attention_layernorm(
        transformer_layer_attention_block_output
    )
    transformer_layer_mlp_output = model._layers[1].mlp_block(transformer_layer_attention_block_output)

    # Final layer norm
    layer_norm_output = model._layers[2](transformer_layer_output)

    # LM Head
    output_logits = model._layers[3](layer_norm_output)

    return_dict["embedding_output"] = embedding_output.activations.clone().detach().cpu()
    return_dict["transformer_layer_output"] = transformer_layer_output.activations.clone().detach().cpu()

    return_dict["transformer_layer_input_layernorm_output"] = (
        transformer_layer_input_layernorm_output.clone().detach().cpu()
    )
    return_dict["transformer_layer_attention_block_output"] = (
        transformer_layer_attention_block_output.clone().detach().cpu()
    )
    return_dict["transformer_layer_post_attention_layernorm_output"] = (
        transformer_layer_post_attention_layernorm_output.clone().detach().cpu()
    )
    return_dict["transformer_layer_mlp_output"] = transformer_layer_mlp_output.clone().detach().cpu()

    return_dict["layer_norm_output"] = layer_norm_output.activations.clone().detach().cpu()
    return_dict["output_logits"] = output_logits.activations.clone().detach().cpu()


@pytest.mark.transformer
def test_backward_compatibility():
    backward_checkpoint_dir = Path(__file__).parents[0] / "files" / "backward_compatibility_checkpoint"
    world_size = 1
    config = TransformerConfig.from_dict(
        {
            "topology": {
                "world_size": world_size,
                "model_parallel_size": 1,
                "data_parallel_size": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
            },
            "transformer_architecture": {
                "vocab_size": 512,
                "sequence_length": 4,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_layers": 1,
            },
        }
    )

    # load ground truth
    ground_truth = torch.load(str(backward_checkpoint_dir / "ground_truth.pt"))

    # get input
    input = ground_truth["input"]

    return_dict_output_logits = dist_launcher(
        run_func=run_forward_pass,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        backward_checkpoint_dir=backward_checkpoint_dir,
        input=input,
    )

    # compare
    def compare(a, b):
        return (a - b).abs().max().item()

    precision = 3.0e-3
    max_difference_in_forward_pass_outputs = [
        (
            "embedding_output",
            compare(
                ground_truth["hidden_states_embedding"],
                return_dict_output_logits["embedding_output"],
            ),
        ),
        (
            "transformer_layer_input_layernorm_output",
            compare(
                ground_truth["hidden_states_input_layernorm"],
                return_dict_output_logits["transformer_layer_input_layernorm_output"],
            ),
        ),
        (
            "transformer_layer_attention_block_output",
            compare(
                ground_truth["hidden_states_attention"],
                return_dict_output_logits["transformer_layer_attention_block_output"],
            ),
        ),
        (
            "transformer_layer_post_attention_layernorm_output",
            compare(
                ground_truth["hidden_states_post_attention_layernorm"],
                return_dict_output_logits["transformer_layer_post_attention_layernorm_output"],
            ),
        ),
        (
            "transformer_layer_mlp_output",
            compare(
                ground_truth["hidden_states_mlp"],
                return_dict_output_logits["transformer_layer_mlp_output"],
            ),
        ),
        (
            "transformer_layer_output",
            compare(
                ground_truth["hidden_states_layer0"],
                return_dict_output_logits["transformer_layer_output"],
            ),
        ),
        (
            "layer_norm_output",
            compare(
                ground_truth["hidden_states_norm"],
                return_dict_output_logits["layer_norm_output"],
            ),
        ),
        (
            "output_logits",
            compare(
                ground_truth["output_logits"],
                return_dict_output_logits["output_logits"],
            ),
        ),
    ]
    assert all(
        [r[1] < precision for r in max_difference_in_forward_pass_outputs]
    ), f"""Functionally different forward pass outputs.\n
        Max differences in outputs: {max_difference_in_forward_pass_outputs}\n
        Affected Components: {[r for r in max_difference_in_forward_pass_outputs if r[1] > precision]}\n
        """
