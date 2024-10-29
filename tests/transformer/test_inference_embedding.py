from pathlib import Path
from typing import Any

import pytest

from scaling.core import MaskedSoftmaxConfig, Topology
from scaling.transformer.context.config import Precision, TransformerConfig
from scaling.transformer.inference import TransformerInferenceModule
from scaling.transformer.model.model import TransformerParallelModule, get_transformer_layer_specs

VOCAB_SIZE = 128000
HIDDEN_SIZE = 256
NUM_ATTENTION_HEADS = 4
NUM_LAYERS = 2
SEQUENCE_LENGTH = 64
PRECISION = Precision.FLOAT16


@pytest.fixture(scope="function")
def inference_model(tmp_path: Path, path_to_files: Path) -> TransformerInferenceModule:
    config_dict: dict[str, Any] = {}
    config_dict["topology"] = {}
    config_dict["topology"]["model_parallel_size"] = 1
    config_dict["topology"]["pipe_parallel_size"] = 1
    config_dict["topology"]["world_size"] = 1
    config_dict["topology"]["global_rank"] = 0
    config_dict["topology"]["local_slot"] = 0
    config_dict["topology"]["micro_batch_size"] = 2
    config_dict["topology"]["gradient_accumulation_steps"] = 1

    config_dict["transformer_architecture"] = {}
    config_dict["transformer_architecture"]["lm_head"] = False
    config_dict["transformer_architecture"]["hidden_size"] = HIDDEN_SIZE
    config_dict["transformer_architecture"]["vocab_size"] = VOCAB_SIZE
    config_dict["transformer_architecture"]["num_layers"] = NUM_LAYERS
    config_dict["transformer_architecture"]["num_attention_heads"] = NUM_ATTENTION_HEADS
    config_dict["transformer_architecture"]["sequence_length"] = SEQUENCE_LENGTH
    config_dict["transformer_architecture"]["precision"] = PRECISION
    config_dict["transformer_architecture"]["masked_softmax"] = MaskedSoftmaxConfig(kernel="torch")

    config = TransformerConfig.from_dict(config_dict)

    topology = Topology(config=config.topology)
    topology.initialize_device()
    layer_specs = get_transformer_layer_specs(architecture_config=config.transformer_architecture, topology=topology)

    parallel_module = TransformerParallelModule(layer_specs=layer_specs, topology=topology)
    (tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
    parallel_module.save_checkpoint(tmp_path / "checkpoint", separate_file_for_parameters=None)
    config.save(tmp_path / "checkpoint" / "config.yml")

    return TransformerInferenceModule.from_checkpoint(
        tmp_path / "checkpoint", vocab_file=path_to_files / "alpha-001-128k.json", devices=(0,)
    )


@pytest.mark.inference
@pytest.mark.parametrize("convert_to_tensor", [True, False])
def test_from_checkpoint(inference_model: TransformerInferenceModule, convert_to_tensor: bool):
    """tests if a transformer checkpoint can be loaded via the from_checkpoint method."""

    result = inference_model.encode(["Hello Darkness", "my old friend"], convert_to_tensor=convert_to_tensor)
    result2 = inference_model.encode(["Hello Darkness my old friend"], convert_to_tensor=convert_to_tensor)
    assert result.shape[0] == 2
    assert result2.shape[0] == 1
    assert result.shape[-1] == result2.shape[-1] == HIDDEN_SIZE
    assert result.shape[1] < result2.shape[1]


@pytest.mark.inference
def test_from_checkpoint_big_batch(inference_model: TransformerInferenceModule):
    result = inference_model.encode(["Hello Darkness"] * 512)
    assert result.shape == (512, 28, 256)
    result = inference_model.encode(["Hello Darkness" * 512], max_length=SEQUENCE_LENGTH)
    assert result.shape == (1, 64, 256)


@pytest.mark.inference
def test_from_checkpoint_instructed(inference_model: TransformerInferenceModule):
    result = inference_model.encode(["Hello Darkness"] * 512, instruction="Instruction is key!")
    assert result.shape == (512, 33, 256)
