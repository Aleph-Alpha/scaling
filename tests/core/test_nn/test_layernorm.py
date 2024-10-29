import pytest
import torch

from scaling.core import LayerNorm, LayerNormConfig, LayerNormOptimizationType


@pytest.mark.nn_rest
@pytest.mark.skipif((not torch.cuda.is_available()), reason="no cuda available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("seq_len", [128 * i for i in range(1, 17)])
@pytest.mark.parametrize("normalized_shape", [768, 2048])
@pytest.mark.parametrize("bitfit_bias_name", [None, "", "test"])
def test_layer_norm(dtype, batch_size, seq_len, normalized_shape, bitfit_bias_name):
    instantiated_implementations = list()
    for layer_norm_optimization_type in LayerNormOptimizationType:
        # Instantiate config to instantiate to be benchmarked module
        config = LayerNormConfig(
            optimization_type=layer_norm_optimization_type,
        )

        # instantiate module dependent on config (e.g. with provider set)
        layer_norm = LayerNorm(
            config=config,
            normalized_shape=normalized_shape,
            device="cuda",
            dtype=dtype,
            bitfit_bias_name=bitfit_bias_name,
        )

        instantiated_implementations.append((layer_norm_optimization_type, layer_norm))

    # seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # input tensor
    # TODO why rescale just to get delta down?
    x = torch.randn(batch_size, seq_len, normalized_shape, device="cuda", dtype=dtype) / 10000

    # wiggle the parameters a little
    weight = torch.randn(
        (normalized_shape,),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn(
        (normalized_shape,),
        dtype=dtype,
        device="cuda",
    )
    for implementation_name, implementation in instantiated_implementations:
        implementation.weight.data = weight.data.clone()
        if bitfit_bias_name is None or bitfit_bias_name == "":
            implementation.bias.data = bias.data.clone()
        else:
            implementation_bias = getattr(implementation, f"bias_{bitfit_bias_name}")
            implementation_bias.data = bias.data.clone()

    assert len(instantiated_implementations) > 0, "no masked softmax implemented"

    # assert that all implementations yield the same result
    ground_truth_name = instantiated_implementations[0][0]
    ground_truth = instantiated_implementations[0][1](x)

    for i in range(1, len(instantiated_implementations)):
        implementation_name, implementation = instantiated_implementations[i]
        compare = implementation(x)
        delta = (compare - ground_truth).abs().max().item()
        if dtype == torch.float32:
            tolerated_delta = 1.0e-6
        elif dtype == torch.float16:
            tolerated_delta = 5.0e-3
        elif dtype == torch.bfloat16:
            tolerated_delta = 5.0e-3
        else:
            raise NotImplementedError
        assert delta < tolerated_delta, (
            f"layernorm implementations for {ground_truth_name} and {implementation_name} "
            f"yield different results with max delta of {delta} for {dtype}"
        )
