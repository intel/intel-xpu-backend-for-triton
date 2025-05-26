"""
Gemm benchmark
============================

This benchmark is come from the Triton tutorial 10-experimental-block-pointer.py
To compare the performance to XeTLA kernel.

"""
from typing import Callable, List, Optional
import os

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark import xetla_kernel
from triton_kernels_benchmark import cutlass_kernel


def get_matmul_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [1, 2, 3]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('large', 32), ('small', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('large', 32), ('small', 64)])
    ]
    return configs


@triton.autotune(
    configs=get_matmul_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # Stride variables
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(tl.float32)

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def get_matmul_batched_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2] for (m, w) in ([('large', 32), ('small', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': 'large'},
            num_stages=s, num_warps=4) for s in [2]
    ]
    return configs


# pylint: disable=unused-argument
@triton.autotune(
    configs=get_matmul_batched_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_block_pointers_batched(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # Stride variables
        stride_az: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bz: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cz: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    bid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz

    a_block_ptr = tl.make_block_ptr(base=a_ptr + offset_a, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(tl.float32)

    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) launches the above kernel.
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    matmul_kernel: Callable,
    matmul_kernel_batched: Callable,
    transpose_a=False,
    transpose_b=False,
):
    a_major, a_minor = -2, -1
    if transpose_a:
        a_major, a_minor = a_minor, a_major
    b_minor, b_major = -2, -1
    if transpose_b:
        b_major, b_minor = b_minor, b_major

    assert a.shape[a_minor] == b.shape[b_minor], 'Incompatible dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    assert b.is_contiguous(), 'Matrix B must be contiguous'
    M, N, K = a.shape[a_major], b.shape[b_major], a.shape[a_minor]
    # Check constraints.
    if len(a.shape) == 3 and len(b.shape) == 3:
        assert a.shape[0] == b.shape[0], 'Incompatible Batch dimension'
        B = a.shape[0]
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            B,
        )
        matmul_kernel_batched[grid](
            a, b, c,  #
            B, M, N, K,  #
            a.stride(0), a.stride(a_major), a.stride(a_minor),  #
            b.stride(0), b.stride(b_minor), b.stride(b_major),  #
            c.stride(0), c.stride(1), c.stride(2))
    elif len(a.shape) == 2 and len(b.shape) == 2:
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(a_major), a.stride(a_minor),  #
            b.stride(b_minor), b.stride(b_major),  #
            c.stride(0), c.stride(1))
    else:
        assert False, 'Input matrixs dimensions mismatch'
    return c


def get_shapes(B, M, N, K, transpose_a, transpose_b):
    a_shape = (M, K)
    if transpose_a:
        a_shape = (K, M)

    b_shape = (K, N)
    if transpose_b:
        b_shape = (N, K)

    if B != 1:
        a_shape = (B, *a_shape)
        b_shape = (B, *b_shape)
    return a_shape, b_shape


X_VALS = [  #
    [1, 1024 * i, 1024 * i, 1024 * i] for i in [1, 2, 4, 8]
] + [  #
    [1, m, n, 4096] for m in [1, 8] for n in [1024, 4096, 6144, 14336, 28672, 128256]
] + [  #
    [1, m, 4096, 14336] for m in [1, 8]
] + [  #
    [1, 1, 13824, 5120],
    [1, 4, 12288, 4096],
    [1, 512, 8192, 8192],
    [1, 512, 8192, 32768],
    [1, 512, 32768, 8192],
    [1, 1024, 8192, 16384],
    [1, 1024, 8192, 28672],
    [1, 3072, 3072, 4096],  # FIXME: Remove this case when gemm_streamk_benchmark can get better performance
    [1, 4096, 8192, 16384],
    [1, 8192, 1024, 16384],
    [1, 8192, 4096, 16384],
    [1, 16384, 1024, 8192],
    [1, 16384, 4096, 8192],
    [1, 16384, 8192, 1024],
    [1, 16384, 8192, 4096],
    [4, 32768, 128, 4096],
    [4, 32768, 4096, 128],
    [32, 4096, 128, 4096],
    [4096, 8, 128, 16384],
    [4096, 8, 16384, 128],
    [1, 8192, 4096, 4096],
]

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    # x_val: (B, M, N, K)
    B, M, N, K = x_val
    # a: (B, M, K) bfloat16
    # b: (B, N, K) bfloat16
    # c: (B, M, N) float32
    # pytorch reference: (B, M, N) float32
    required_memory = B * M * K * 2 + B * N * K * 2 + 2 * B * M * N * 4
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    transpose_a=False,
    transpose_b=False,
    matmul_kernel=matmul_kernel_with_block_pointers,
    matmul_kernel_batched=matmul_kernel_with_block_pointers_batched,
    plot_name='matmul-performance',
):
    """
    Returns a Mark object containing a Benchmark object constructed at runtime and parameterized by the provided option values.
    The benchmark can then be executed by calling the :code:`.run` method on the return value.
    """
    supported_providers = {
        'triton': 'Triton',
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    # Benchmark Performance
    # pylint: disable=too-many-branches
    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['B', 'M', 'N', 'K'],
            # different possible values for `x_name`
            x_vals=X_VALS,
            line_arg='provider',
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=list(providers.keys()),
            # label name for the lines
            line_names=list(providers.values()),
            # line styles
            styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
            ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
            plot_name=plot_name,
            # name for the plot. Used also as a file name for saving the plot.
            args={},
        ))
    def benchmark(B, M, N, K, provider):
        print("Benchmarking with parameters: ", B, M, N, K, provider)
        a_shape, b_shape = get_shapes(B, M, N, K, transpose_a=transpose_a, transpose_b=transpose_b)

        torch.manual_seed(0)
        a = torch.rand(a_shape, device='xpu', dtype=torch.bfloat16)
        b = torch.rand(b_shape, device='xpu', dtype=torch.bfloat16)

        quantiles = [0.5, 0.0, 1.0]

        torch_a = a
        if transpose_a:
            torch_a = torch.transpose(torch_a, -2, -1)

        torch_b = b
        if transpose_b:
            torch_b = torch.transpose(torch_b, -2, -1)

        if provider == 'onednn':
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                lambda: torch.matmul(torch_a, torch_b),
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'triton':
            if len(a.shape) != len(b.shape):
                raise AssertionError(f'Incompatible sizes {len(a.shape)} and {len(b.shape)}', )
            if len(a.shape) == 3:
                c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
            elif len(a.shape) == 2:
                c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
            else:
                raise AssertionError(f'Unexpected shape of length {len(a.shape)}')
            triton_fn = lambda: matmul(
                a,
                b,
                c,
                matmul_kernel=matmul_kernel,
                matmul_kernel_batched=matmul_kernel_batched,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
            torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)
            rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='triton to torch')
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'xetla':
            if B == 1:
                c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                cnt = torch.zeros((M, N), device='xpu', dtype=torch.int32)
            else:
                c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                cnt = torch.zeros((B, M, N), device='xpu', dtype=torch.int32)
            name = f'gemm_shape_{B}_{M}_{K}_{N}'
            # FIXME: Use gemm_streamk_benchmark.py when Triton streamk can get
            # better performance.
            if (B, M, N, K) == (1, 3072, 3072, 4096):
                name = 'gemm_streamk_shape_3072_4096_3072'
            func = getattr(xetla_kernel, name)

            def xetla_func_with_acc_allocation():
                # allocating `acc` matrix on every function call, to be as similar as
                # possible to the triton kernel, which also does this on every call.
                if B == 1:
                    acc = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                else:
                    acc = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                return func(a, b, c, acc, cnt)

            xetla_fn = xetla_func_with_acc_allocation
            torch_fn = lambda: torch.matmul(a, b).to(torch.float32)

            # benchmark_suite.assert_close(xetla_fn, torch_fn, atol=1e-4, rtol=1.0, err_msg='xetla to torch')
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                xetla_fn,
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'cutlass':
            name = 'gemm'
            func = getattr(cutlass_kernel, name)

            def cutlass_invoker():
                if B == 1:
                    c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                else:
                    c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                func(a, b, c, M, N, K, B)
                return c

            cutlass_fn = cutlass_invoker
            torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)

            rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
            benchmark_suite.assert_close(cutlass_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='cutlass to torch')
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                cutlass_fn,
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        tflops = lambda ms: 2 * B * M * N * K * (1e-12) / (ms * 1e-3)
        gbps = lambda ms: B * (2 * (M * K + K * N) + 4.0 * (M * N)) * (1e-9) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    _benchmark = get_benchmark(
        transpose_a=(os.getenv('TRANSPOSE_A', '0') == '1'),
        transpose_b=(os.getenv('TRANSPOSE_B', '0') == '1'),
    )
    _benchmark.run(show_plots=False, print_data=True)
