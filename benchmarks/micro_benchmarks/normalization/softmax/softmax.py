"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices: those whose rows can fit in
the GPU's SRAM.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

This code is developed by referring softmax tutorial from openAI Triton github repo.
"""

# %%
# Motivations
# -----------
#
# Custom GPU kernels for elementwise additions are educationally valuable but won't get you very far in practice.
# Let us consider instead the case of a simple (numerically stabilized) softmax operation:

import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl


def torch_softmax_inf(x):
    y = torch.softmax(x, axis=-1)
    return y


def torch_softmax_train(x, dy):
    y = torch.softmax(x, axis=-1)
    y.backward(dy, retain_graph=True)
    return x.grad


def triton_softmax_inf(x):
    y = Softmax.apply(x)
    return y


def triton_softmax_train(x, dy):
    y = Softmax.apply(x)
    y.backward(dy, retain_graph=True)
    return x.grad


@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


# %%
# When implemented naively in PyTorch, computing :code:`y = naive_softmax(x)` for :math:`x \in R^{M \times N}`
# requires reading :math:`5MN + 2M` elements from DRAM and writing back :math:`3MN + 2M` elements.
# This is obviously wasteful; we'd prefer to have a custom "fused" kernel that only reads
# X once and does all the necessary computations on-chip.
# Doing so would require reading and writing back only :math:`MN` bytes, so we could
# expect a theoretical speed-up of ~4x (i.e., :math:`(8MN + 4M) / 2MN`).
# The `torch.jit.script` flags aims to perform this kind of "kernel fusion" automatically
# but, as we will see later, it is still far from ideal.

# %%
# Compute Kernel
# --------------
#
# Our softmax kernel works as follows: each program loads a row of the input matrix X,
# normalizes it and writes back the result to the output Y.
#
# Note that one important limitation of Triton is that each block must have a
# power-of-two number of elements, so we need to internally "pad" each row and guard the
# memory operations properly if we want to handle any possible input shapes:


@triton.jit
def _softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def _softmax_backward_kernel(grad_input, grad_output, output, grad_input_stride, grad_out_stride, output_row_stride,
                             n_cols, BLOCK_SIZE: tl.constexpr):
    # Parallelization across rows
    row_idx = tl.program_id(0)

    # Memory pointer calculations
    row_start_ptr = grad_input + row_idx * grad_input_stride
    grad_output_row_start_ptr = grad_output + row_idx * grad_out_stride
    output_row_start_ptr = output + row_idx * output_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Memmory addresses of all the elements we want to load
    grad_output_ptrs = grad_output_row_start_ptr + col_offsets
    output_ptrs = output_row_start_ptr + col_offsets

    # Load relevant data
    o = tl.load(output_ptrs, mask=col_offsets < n_cols)
    g = tl.load(grad_output_ptrs, mask=col_offsets < n_cols)

    # Using cross-entropy loss
    # Step1: Compute intermediate sum used for gradient
    s = tl.sum(g * o, 0)

    # Step1: Compute the gradients
    grad_input = o * (g - s)

    grad_input_ptrs = row_start_ptr + col_offsets
    tl.store(grad_input_ptrs, grad_input, mask=col_offsets < n_cols)


# %%
# We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.


class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        # The block size is the smallest power of two greater than the number of columns in `x`
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        # Another trick we can use is to ask the compiler to use more threads per row by
        # increasing the number of warps (`num_warps`) over which each row is distributed.
        # You will see in the next tutorial how to auto-tune this value in a more natural
        # way so you don't have to come up with manual heuristics yourself.
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        # Allocate output
        y = torch.empty_like(x)
        # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
        # f the input matrix
        _softmax_kernel[(n_rows, )](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (out, ) = ctx.saved_tensors
        n_rows, n_cols = out.shape

        # The block size is the smallest power of two greater than the number of columns in `x`
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # torch.zeros is measurably slower, we'll zero out in the kernel
        grad_in = torch.empty_like(out)

        # Make sure that the tensor are contiguous
        grad_in, grad_out, out = map(lambda x: x.contiguous(), [grad_in, grad_out, out])
        _softmax_backward_kernel[(n_rows, )](
            grad_in,
            grad_out,
            out,
            grad_in.stride(0),
            grad_out.stride(0),
            out.stride(0),
            n_cols,
            BLOCK_SIZE,
        )
        return grad_in.reshape_as(grad_out)


# %%
# Unit Test
# ---------

# %%
# We make sure that we test our kernel on a matrix with an irregular number of rows and columns.
# This will allow us to verify that our padding mechanism works.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton-inf", "triton-train", "torch-inf", "torch_train"],
        line_names=["triton-inf", "triton-train", "torch-inf", "torch_train"],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096, "dtype": torch.float16},
    ))
def benchmark(M, N, provider, dtype):

    # create data
    x = torch.randn(M, N, device="xpu", dtype=dtype, requires_grad=True)
    quantiles = [0.5, 0.2, 0.8]
    dy = .1 * torch.randn_like(x)
    if provider == 'torch-inf':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_softmax_inf(x), quantiles=quantiles)
    if provider == 'torch_train':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_softmax_train(x, dy), quantiles=quantiles)
    if provider == 'triton-inf':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax_inf(x), quantiles=quantiles)
    if provider == 'triton-train':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax_train(x, dy), quantiles=quantiles)

    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
