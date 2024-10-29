import pytest
import torch

from scaling.core.fp8 import fp8_linear


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 9), reason="CUDA device not FP8-capable.")
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype_base", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "dtypes_fwd",
    [
        None,
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize(
    "dtypes_grad_input",
    [
        None,
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize(
    "dtypes_grad_weight",
    [
        None,
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
)
def test_float8_linear(
    use_bias: bool,
    dtype_base: torch.dtype,
    dtypes_fwd: tuple[torch.dtype, torch.dtype],
    dtypes_grad_input: tuple[torch.dtype, torch.dtype],
    dtypes_grad_weight: tuple[torch.dtype, torch.dtype],
):
    torch.manual_seed(42)
    input = torch.randn(16, 32, dtype=dtype_base, device="cuda")
    weight = torch.empty(64, 32, dtype=dtype_base, device="cuda", requires_grad=True)
    torch.nn.init.xavier_normal_(weight)
    bias = torch.randn(64, dtype=dtype_base, device="cuda") if use_bias else None

    output_baseline = torch.nn.functional.linear(input, weight, bias)
    output_baseline.sum().backward()
    assert isinstance(weight.grad, torch.Tensor)
    grad_baseline = weight.grad.clone()
    weight.grad = None

    output_fp8 = fp8_linear(input, weight, bias, dtypes_fwd, dtypes_grad_input, dtypes_grad_weight)
    output_fp8.sum().backward()
    assert isinstance(weight.grad, torch.Tensor)
    grad_fp8 = weight.grad.clone()
    weight.grad = None

    assert output_fp8.dtype is output_baseline.dtype
    assert (output_fp8 - output_baseline).abs().sum() / output_baseline.abs().sum() < 1e-1

    assert grad_fp8.dtype is grad_baseline.dtype
    assert (grad_fp8 - grad_baseline).abs().sum() / grad_baseline.abs().sum() < 1e-1
