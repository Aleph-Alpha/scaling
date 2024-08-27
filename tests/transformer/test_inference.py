from pathlib import Path
from typing import Any, Dict, Sequence

import pytest
import torch

from scaling.core import MaskedSoftmaxConfig, MaskedSoftmaxKernel, Topology
from scaling.core.nn.parallel_module.inference_module import RecorderSetting
from scaling.transformer.context.config import Precision, TransformerArchitectureConfig, TransformerConfig
from scaling.transformer.inference import TransformerInferenceModule
from scaling.transformer.inference.sample import sample_temperature, top_k_transform, top_p_transform
from scaling.transformer.model.layers.base import TransformerLayerIO
from scaling.transformer.model.model import TransformerParallelModule, get_transformer_layer_specs
from scaling.transformer.tokenizer.tokenizer import Tokenizer

VOCAB_SIZE = 128000
HIDDEN_SIZE = 256
NUM_ATTENTION_HEADS = 4
NUM_LAYERS = 2
SEQUENCE_LENGTH = 64
PRECISION = Precision.FLOAT16

pytestmark = [
    pytest.mark.parametrize("input_text, input_tokens", [("Once upon a time", None), (None, [1, 2, 3, 4])]),
    pytest.mark.parametrize("devices", [(0,), (0, 1)]),
    pytest.mark.parametrize("masked_softmax_kernel", [MaskedSoftmaxKernel.TORCH, MaskedSoftmaxKernel.FLASH_ATTENTION]),
]


@pytest.fixture()
def inference_module(masked_softmax_kernel: MaskedSoftmaxKernel, devices: Sequence[int]):
    torch.manual_seed(42)
    layer_specs = get_transformer_layer_specs(
        architecture_config=TransformerArchitectureConfig(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            masked_softmax=MaskedSoftmaxConfig(kernel=masked_softmax_kernel),
            precision=PRECISION,
        )
    )
    tokenizer = Tokenizer.from_file(str(Path(__file__).parent / "files" / "alpha-001-128k.json"))
    return TransformerInferenceModule(layer_specs=layer_specs, tokenizer=tokenizer, devices=devices)


def test_generate(inference_module: TransformerInferenceModule, input_text: str | None, input_tokens: list[int] | None):
    """tests generate method and whether cached and uncached inference give the same results"""

    result_1 = inference_module.generate(
        max_tokens=10,
        input_text=input_text,
        input_tokens=input_tokens,
        stop_tokens=(0,),
        use_cache=True,
    )

    result_2 = inference_module.generate(
        max_tokens=10,
        input_text=input_text,
        input_tokens=input_tokens,
        stop_tokens=(0,),
        use_cache=False,
    )

    if input_text is not None:
        assert result_1.completion_text is not None
        assert result_2.completion_text is not None
    assert tuple(result_1.completion_tokens) == tuple(result_2.completion_tokens)
    # on RTX 3090, this passes with rtol=e-5, atol=e-5
    assert torch.allclose(result_1.completion_logits, result_2.completion_logits, rtol=1.0e-2, atol=1.0e-2)

    def sample_fn(x: torch.Tensor):
        return sample_temperature(top_p_transform(top_k_transform(x)))

    inference_module.generate(
        max_tokens=10, input_text=input_text, input_tokens=input_tokens, sample_fn=sample_fn, stop_tokens=(0,)
    )


def test_logits(inference_module: TransformerInferenceModule, input_text: str | None, input_tokens: list[int] | None):
    "tests simple forward pass"
    logits = inference_module.logits(input_text=input_text, input_tokens=input_tokens)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, VOCAB_SIZE)


@pytest.mark.parametrize("recorder_scenario", [1, 2, 3])
def test_logits_with_hidden_states_recorder(
    inference_module: TransformerInferenceModule,
    input_text: str | None,
    input_tokens: list[int] | None,
    recorder_scenario: int,
):
    if recorder_scenario == 1:
        # get all layer outputs
        recorder_settings_per_layer = {k: RecorderSetting() for k in range(len(inference_module._layers))}
    elif recorder_scenario == 2:
        # get layer output from last transformer layer only
        recorder_settings_per_layer = {2: RecorderSetting()}
    elif recorder_scenario == 3:
        # get all self attention outputs from transformer layers
        recorder_settings_per_layer = {
            1: RecorderSetting(include_modules=("self_attention",)),
            2: RecorderSetting(include_modules=("self_attention",)),
        }

    logits, recorder_results = inference_module.logits_with_hidden_state_recorder(
        input_text=input_text, input_tokens=input_tokens, recorder_settings_per_layer=recorder_settings_per_layer
    )

    if recorder_scenario == 1:
        assert set(recorder_results.keys()) == {0, 1, 2, 3, 4}
    elif recorder_scenario == 2:
        assert set(recorder_results.keys()) == {2}
    elif recorder_scenario == 3:
        set(recorder_results.keys()) == {1, 2}

    for layer_index, layer_dict in recorder_results.items():
        if recorder_scenario in (1, 2):
            assert layer_dict.keys() == {""}
            for k, v in layer_dict.items():
                assert isinstance(v, TransformerLayerIO)
        elif recorder_scenario == 3:
            assert layer_dict.keys() == {"self_attention"}
            for k, v in layer_dict.items():
                assert isinstance(v, torch.Tensor)


def test_from_checkpoint(
    tmp_path: Path,
    devices: Sequence[int],
    masked_softmax_kernel: MaskedSoftmaxKernel,
    input_text: str | None,
    input_tokens: list[int] | None,
):
    """tests if a transformer checkpoint can be loaded via the from_checkpoint method."""

    config_dict: Dict[str, Any] = dict()

    config_dict["topology"] = dict()
    config_dict["topology"]["model_parallel_size"] = 1
    config_dict["topology"]["pipe_parallel_size"] = 1
    config_dict["topology"]["world_size"] = 1
    config_dict["topology"]["global_rank"] = 0
    config_dict["topology"]["local_slot"] = 0
    config_dict["topology"]["micro_batch_size"] = 2
    config_dict["topology"]["gradient_accumulation_steps"] = 1

    config_dict["transformer_architecture"] = dict()
    config_dict["transformer_architecture"]["hidden_size"] = HIDDEN_SIZE
    config_dict["transformer_architecture"]["vocab_size"] = VOCAB_SIZE
    config_dict["transformer_architecture"]["num_layers"] = NUM_LAYERS
    config_dict["transformer_architecture"]["num_attention_heads"] = NUM_ATTENTION_HEADS
    config_dict["transformer_architecture"]["sequence_length"] = SEQUENCE_LENGTH
    config_dict["transformer_architecture"]["precision"] = PRECISION
    config_dict["transformer_architecture"]["masked_softmax"] = MaskedSoftmaxConfig(kernel=masked_softmax_kernel)

    config = TransformerConfig.from_dict(config_dict)

    topology = Topology(config=config.topology)
    topology.initialize_device()
    layer_specs = get_transformer_layer_specs(architecture_config=config.transformer_architecture, topology=topology)

    parallel_module = TransformerParallelModule(layer_specs=layer_specs, topology=topology)
    (tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
    parallel_module.save_checkpoint(tmp_path / "checkpoint", separate_file_for_parameters=None)
    config.save(tmp_path / "checkpoint" / "config.yml")

    inference_model = TransformerInferenceModule.from_checkpoint(
        tmp_path / "checkpoint", vocab_file=Path(__file__).parent / "files" / "alpha-001-128k.json", devices=devices
    )

    inference_model.generate(max_tokens=10, input_text=input_text, input_tokens=input_tokens, stop_tokens=(0,))
