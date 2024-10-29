from typing import Any

import torch

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max


def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor:
    assert float8_dtype in FP8_DTYPES, f"Expected an fp8 dtype, received {float8_dtype=}."
    if x.dtype is float8_dtype:
        return x
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)


def is_row_major(stride: tuple[int, ...]) -> bool:
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1


def make_contiguous_for_matmul(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_row_major(a.stride()):
        a = a.contiguous()
    if is_row_major(b.stride()):
        b = b.t().contiguous().t()
    return a, b


def matmul(a: torch.Tensor, b: torch.Tensor, dtypes: tuple[torch.dtype, torch.dtype] | None) -> torch.Tensor:
    if dtypes is None:
        output = torch.mm(a, b)
    else:
        out_dtype = a.dtype
        a = to_fp8_saturated(a, dtypes[0])
        b = to_fp8_saturated(b, dtypes[1])
        a, b = make_contiguous_for_matmul(a, b)
        output, _ = torch._scaled_mm(a, b, out_dtype=out_dtype)
    return output


class FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        dtypes_forward: tuple[torch.dtype, torch.dtype] | None = None,
        dtypes_grad_input: tuple[torch.dtype, torch.dtype] | None = None,
        dtypes_grad_weight: tuple[torch.dtype, torch.dtype] | None = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.dtypes_grad_input = dtypes_grad_input
        ctx.dtypes_grad_weight = dtypes_grad_weight
        output = matmul(input, weight, dtypes_forward)
        return output

    @staticmethod
    def backward(  # type: ignore[override] # noqa
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None]:
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = matmul(grad_output, weight.t(), ctx.dtypes_grad_input)
        if ctx.needs_input_grad[1]:
            grad_weight = matmul(input.t(), grad_output, ctx.dtypes_grad_weight)
        return grad_input, grad_weight, None, None, None, None, None


def fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtypes_forward: tuple[torch.dtype, torch.dtype] | None = (torch.float8_e4m3fn, torch.float8_e4m3fn),
    dtypes_grad_input: tuple[torch.dtype, torch.dtype] | None = (torch.float8_e5m2, torch.float8_e4m3fn),
    dtypes_grad_weight: tuple[torch.dtype, torch.dtype] | None = (torch.float8_e4m3fn, torch.float8_e5m2),
) -> torch.Tensor:
    """Linear operation with custom FP8 dtypes for each matmul in forward and backward.

    No FP8 casting is performed for a matmul if the corresponding `dtypes_` is set to `None`.
    The output dtype is always set to the dtype of `input`.

    Args:
        input: Input tensor.
        weight: Weight tensor.
        bias: Optional bias tensor.
        dtypes_fwd: Data types for the forward matmul (x, w).
        dtypes_grad_input: Data types for the grad input matmul (grad_y, w).
        dtypes_grad_weight: Data types for the grad weight matmul (x, grad_y).

    Returns:
        The output tensor of the linear operation.
    """
    input_flat = input.view(-1, input.size(-1))
    out = FP8Linear.apply(
        input_flat,
        weight.t(),
        dtypes_forward,
        dtypes_grad_input,
        dtypes_grad_weight,
    )
    if bias is not None:
        out += bias
    return out.view(*input.size()[:-1], out.size(-1))
