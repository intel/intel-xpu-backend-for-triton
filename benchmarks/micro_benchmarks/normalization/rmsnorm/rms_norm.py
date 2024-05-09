import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl

from typing import Any
from torch.profiler import profile, record_function, ProfilerActivity


class util:

    @staticmethod
    def dtype(input):
        if input == torch.float32:
            return tl.float32
        elif input == torch.float16:
            return tl.float16
        elif input == torch.bfloat16:
            return tl.bfloat16
        elif input == torch.int64:
            return tl.int64
        else:
            raise ValueError(f"Unable to convert the given input: '{input}'.")

    @staticmethod
    def size_and_stride(input: torch.Tensor, dim: int):
        if input.dim() == 2:
            if dim == 0:
                x_size, y_size = input.shape
                y_stride = input.stride(1)
                x_stride = input.stride(0)
            else:
                y_size, x_size = input.shape
                y_stride = input.stride(0)
                x_stride = input.stride(1)

            return y_size, x_size, y_stride, x_stride
        elif input.dim() == 3:
            if dim == 0:
                z_size, y_size, x_size = input.shape[0], input.shape[1], input.shape[2]
                z_stride, y_stride, x_stride = input.stride(0), input.stride(1), input.stride(2)
            elif dim == 1:
                z_size, y_size, x_size = input.shape[1], input.shape[0], input.shape[2]
                z_stride, y_stride, x_stride = input.stride(1), input.stride(0), input.stride(2)
            else:
                z_size, y_size, x_size = input.shape[2], input.shape[0], input.shape[1]
                z_stride, y_stride, x_stride = input.stride(2), input.stride(0), input.stride(1)

            return z_size, y_size, x_size, z_stride, y_stride, x_stride
        else:
            raise ValueError(f"{dim} is not supported.")


@triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
@triton.jit
def rms_norm_forward(
    output_ptr: tl.tensor,
    rms_ptr: tl.tensor,
    input_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    y_stride: tl.int32,
    x_stride: tl.int32,
    partial_size: tl.constexpr,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    eps: tl.float32,
    dtype: tl.constexpr,
    x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr,
):
    y_offset = tl.program_id(0)

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    rms_block_ptr = tl.make_block_ptr(
        rms_ptr,
        shape=(y_size, ),
        strides=(1, ),
        offsets=(y_offset, ),
        block_shape=(1, ),
        order=(0, ),
    )
    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(x_size, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(x_block_size, ),
        order=(0, ),
    )

    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1, ))
    else:
        input = tl.load(input_block_ptr)

    if x_block_size != partial_size:
        condition = tl.arange(0, x_block_size) < partial_size
        partial_input = tl.where(condition, input, 0)
    else:
        partial_input = input

    rms = tl.math.sqrt(tl.sum(partial_input * partial_input / partial_size, 1))
    norm = input / (rms + eps)

    if require_x_boundary_check:
        weight = tl.load(weight_block_ptr, boundary_check=(0, ))
    else:
        weight = tl.load(weight_block_ptr)

    output = norm * weight

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            bias_ptr,
            shape=(1, x_size),
            strides=(x_stride, 1),
            offsets=(0, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        if require_x_boundary_check:
            bias = tl.load(bias_block_ptr, boundary_check=(1, ))
        else:
            bias = tl.load(bias_block_ptr)

        output += bias

    tl.store(rms_block_ptr, rms.to(dtype))

    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1, ))
    else:
        tl.store(output_block_ptr, output.to(dtype))


@triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
@triton.jit
def rms_norm_backward(
    grad_input_ptr: tl.tensor,
    grad_weight_staging: tl.tensor,
    grad_output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    y_stride: tl.int32,
    x_stride: tl.int32,
    rms_ptr: tl.tensor,
    partial_size: tl.constexpr,
    weight_ptr: tl.tensor,
    eps: tl.float32,
    dtype: tl.constexpr,
    x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr,
):
    y_offset = tl.program_id(0)

    grad_input_block_ptr = tl.make_block_ptr(
        grad_input_ptr,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    grad_weight_staging_block_ptr = tl.make_block_ptr(
        grad_weight_staging,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(1, x_block_size),
        order=(1, 0),
    )
    rms_block_ptr = tl.make_block_ptr(
        rms_ptr,
        shape=(y_size, ),
        strides=(1, ),
        offsets=(y_offset, ),
        block_shape=(1, ),
        order=(0, ),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(x_size, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(x_block_size, ),
        order=(0, ),
    )

    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1, ))
        input = tl.load(input_block_ptr, boundary_check=(1, ))
    else:
        grad_output = tl.load(grad_output_block_ptr)
        input = tl.load(input_block_ptr)

    rms = tl.load(rms_block_ptr)

    if require_x_boundary_check:
        weight = tl.load(weight_block_ptr, boundary_check=(0, ))
    else:
        weight = tl.load(weight_block_ptr)

    grad_norm = grad_output * weight
    norm = input / (rms + eps)
    grad_weight = grad_output * norm

    if require_x_boundary_check:
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype), boundary_check=(1, ))
    else:
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))

    grad_rms = grad_norm * -input / (rms * rms + eps)

    if require_x_boundary_check:
        condition = tl.arange(0, x_block_size) < x_size
        grad_rms = tl.where(condition, grad_rms, 0.0)

    grad_rms = tl.sum(grad_rms, 1)
    grad_mean_square = grad_rms / (2 * rms)
    grad_partial_input = 2 * input * grad_mean_square / partial_size

    if x_block_size != partial_size:
        condition = tl.arange(0, x_block_size) < partial_size
        grad_partial_input = tl.where(condition, grad_partial_input, 0)

    grad_input = (grad_norm / (rms + eps)) + grad_partial_input

    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1, ))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))


def rms_norm_ref(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-08):
    y_size, x_size = input.shape

    if p < 0.0 or p > 1.0:
        norm = input.norm(2, dim=-1, keepdim=True)
        partial_size = x_size
    else:
        partial_size = int(x_size * p)
        partial_input, _ = torch.split(input, [partial_size, x_size - partial_size], dim=-1)
        norm = partial_input.norm(2, dim=-1, keepdim=True)

    rms = norm * partial_size**(-1.0 / 2)
    output = input / (rms + eps)

    if bias is not None:
        return weight * output + bias

    return weight * output


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, p, weight, bias, eps = args
        output, rms = RMSNorm.__forward(input, p, weight, bias, eps)
        ctx.save_for_backward(input, rms, weight, bias)
        ctx.p = p
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output, ) = grad_outputs
        input, rms, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = RMSNorm.__backward(grad_output, input, ctx.p, rms, weight, bias, ctx.eps)

        return grad_input, None, grad_weight, grad_bias, None

    @staticmethod
    def __forward(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)

        output = torch.empty_like(input)
        rms = torch.empty(y_size, **factory_kwargs)

        grid = lambda meta: (y_size, )
        rms_norm_forward[grid](
            output,
            rms,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            x_size if p < 0.0 or p > 1.0 else x_size * p,
            weight,
            bias,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )

        return output, rms

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        p: float,
        rms: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)
        grad_input = torch.empty_like(grad_output)
        grad_weight_staging = torch.empty((y_size, x_size), **factory_kwargs)

        grid = lambda meta: (y_size, )
        rms_norm_backward[grid](
            grad_input,
            grad_weight_staging,
            grad_output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            rms,
            x_size if p < 0.0 or p > 1.0 else x_size * p,
            weight,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )

        grad_weight = torch.sum(grad_weight_staging, 0)

        if bias is not None:
            grad_bias = torch.sum(grad_output, 0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias


def rms_norm(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-08):
    """
    Initializes the model
    """
    return RMSNorm.apply(input.view(-1, input.shape[-1]), p, weight, bias, eps).view(input.shape)


def test_forward(y_size, x_size, p, device='xpu'):
    input = torch.randn(y_size, x_size, device=device)
    weight = torch.randn(x_size, device=device)
    assert torch.allclose(rms_norm_ref(input, p, weight), rms_norm(input, p, weight), atol=1e-2, rtol=0)
    bias = torch.randn(x_size, device=device)
    assert torch.allclose(rms_norm_ref(input, p, weight, bias), rms_norm(input, p, weight, bias), atol=1e-2, rtol=0)


def test_backward(y_size, x_size, p, device='xpu'):
    input = torch.randn((y_size, x_size), device=device)
    weight = torch.randn(x_size, device=device)
    grad_output = torch.randn(y_size, x_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, p, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(rms_norm_ref)
    (a, b) = train(rms_norm)

    assert torch.allclose(x, a, atol=1e-2, rtol=0)
    assert torch.allclose(y, b, atol=1e-2, rtol=0)

    bias = torch.randn(x_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, p, j, k).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(rms_norm_ref)
    (a, b, c) = train(rms_norm)

    assert torch.allclose(x, a, atol=1e-2, rtol=0)
    assert torch.allclose(y, b, atol=1e-2, rtol=0)
    assert torch.allclose(z, c, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[256, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def benchmark(M, N, dtype, provider, mode='backeard', eps=1e-5):
    # create data

    input = torch.randn((N, M), device="xpu", dtype=dtype)
    weight = torch.randn(M, device="xpu", dtype=dtype)
    grad_output = torch.randn(N, M, device="xpu", dtype=dtype)
    bias = torch.randn(M, device="xpu", dtype=dtype)
    p = 1.0
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        y_fwd = rms_norm
    if provider == 'torch':
        y_fwd = rms_norm_ref

    # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y_fwd(input, p, weight, bias), quantiles=quantiles)
        gbps = lambda ms: 2 * input.numel() * input.element_size() / ms * 1e-6
    # backward pass
    else:
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        y = y_fwd(i, p, j, k)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(grad_output, retain_graph=True),
                                                     quantiles=quantiles)
        gbps = lambda ms: 3 * input.numel() * input.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    benchmark.run(print_data=True)
