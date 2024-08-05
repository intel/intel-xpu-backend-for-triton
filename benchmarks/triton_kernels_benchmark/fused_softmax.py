"""
Fused Softmax
=============

This benchmark is come from the Triton tutorial 02-fused-softmax
To compare the performance to XeTLA kernel.

"""

import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
from triton.runtime import driver

import triton_kernels_benchmark
import triton_kernels_benchmark.xetla_kernel as xetla_kernel

benchmark_suit = triton_kernels_benchmark  # triton.testing


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


@triton.autotune(
    configs=[
        triton.Config({"threads_per_warp": 32}, num_warps=32),
        triton.Config({"threads_per_warp": 32}, num_warps=16),
        triton.Config({"threads_per_warp": 32}, num_warps=8),
        triton.Config({"threads_per_warp": 32}, num_warps=4),
        triton.Config({"threads_per_warp": 16}, num_warps=64),
        triton.Config({"threads_per_warp": 16}, num_warps=32),
        triton.Config({"threads_per_warp": 16}, num_warps=16),
        triton.Config({"threads_per_warp": 16}, num_warps=8),
        triton.Config({"threads_per_warp": 16}, num_warps=4),
    ],
    key=['BLOCK_SIZE_X', 'BLOCK_SIZE_Y'],
)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE_X: tl.constexpr,
                   BLOCK_SIZE_Y: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0) * BLOCK_SIZE_Y
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE_X)
    row_offsets = tl.arange(0, BLOCK_SIZE_Y)
    offsets = col_offsets[None, :] + row_offsets[:, None] * input_row_stride
    input_ptrs = row_start_ptr + offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets[None, :] < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


device = torch.xpu.current_device()
properties = driver.active.utils.get_device_properties(device)
MAX_WORK_GROUP_SIZE = properties["max_work_group_size"]


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE_X = triton.next_power_of_2(n_cols)
    BLOCK_SIZE_Y = MAX_WORK_GROUP_SIZE // BLOCK_SIZE_X
    BLOCK_SIZE_Y = BLOCK_SIZE_Y if BLOCK_SIZE_Y > 0 else 1

    # Allocate output
    y = torch.empty_like(x)
    # Create a number of persistent programs.
    softmax_kernel[(n_rows // BLOCK_SIZE_Y, )](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE_X=BLOCK_SIZE_X,
                                               BLOCK_SIZE_Y=BLOCK_SIZE_Y)
    return y


@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[256, 1024, 2048, 4096, 1024 * 8, 1024 * 16, 1024 * 32],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            # 'torch-native',
            # 'torch-jit',
            'xetla',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            # "Torch (native)",
            # "Torch (jit)",
            "XeTLA",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--'), ('black', ':')],  # line styles
        ylabel=["GB/s", "TFlops"],  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='xpu', dtype=torch.bfloat16)
    quantiles = [0.5, 0.0, 1.0]
    if provider == 'torch-native':
        ms, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles,
                                                               warmup=10, rep=10)
    if provider == 'triton':
        triton_fn = lambda: softmax(x)
        torch_fn = lambda: torch.softmax(x, axis=-1)
        benchmark_suit.assert_close(triton_fn(), torch_fn(), err_msg="triton to torch")
        ms, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, quantiles=quantiles, warmup=10, rep=10)

    if provider == 'torch-jit':
        ms, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(lambda: naive_softmax(x), quantiles=quantiles, warmup=10,
                                                               rep=10)

    if provider == 'xetla':
        name = "softmax_shape_{}_{}".format(M, N)
        func = getattr(xetla_kernel, name)
        xetla_fn = lambda: func(x, 0)
        torch_fn = lambda: torch.softmax(x, axis=-1)
        # benchmark_suit.assert_close(xetla_fn(), torch_fn(), err_msg="xetla to torch")
        ms, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(xetla_fn, quantiles=quantiles, warmup=10, rep=10)

    gbps = lambda mean: 2 * x.nelement() * x.element_size() * 1e-9 / (mean * 1e-3)
    tflops = lambda mean: 4 * x.nelement() * 1e-12 / (mean * 1e-3
                                                      )  # reduce-max, reduce-sum, elem-wise sub, elem-wise div
    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)
