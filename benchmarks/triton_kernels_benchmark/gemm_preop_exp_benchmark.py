"""
Gemm + PreOp (exp) benchmark
============================

This benchmark is modified from gemm_benchmark.py to include an exponential
operation on the input matrix A.

"""

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suit


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=3, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_tensor_descriptors(
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

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        a = a.to(tl.float32)
        a = tl.math.exp(a)
        a = a.to(tl.bfloat16)
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator += tl.dot(a, b)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)

    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


# pylint: disable=unused-argument
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=3, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': 'large'},
            num_stages=2, num_warps=32),
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': 'large'},
            num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_tensor_descriptors_batched(
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

    a_desc = tl.make_tensor_descriptor(base=a_ptr + offset_a, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        a = a.to(tl.float32)
        a = tl.math.exp(a)
        a = a.to(tl.bfloat16)
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator += tl.dot(a, b)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)

    offset_c = bid.to(tl.int64) * stride_cz
    c_desc = tl.make_tensor_descriptor(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) launches the above kernel.
def matmul(a, b, c):
    # Check constraints.
    if len(a.shape) == 3 and len(b.shape) == 3:
        assert a.shape[0] == b.shape[0], 'Incompatible Batch dimension'
        assert a.shape[2] == b.shape[1], 'Incompatible dimensions'
        assert a.is_contiguous(), 'Matrix A must be contiguous'
        assert b.is_contiguous(), 'Matrix B must be contiguous'
        B, M, K = a.shape
        B, K, N = b.shape
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            B,
        )
        matmul_kernel_with_tensor_descriptors_batched[grid](
            a, b, c,  #
            B, M, N, K,  #
            a.stride(0), a.stride(1), a.stride(2),  #
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), c.stride(2))
    elif len(a.shape) == 2 and len(b.shape) == 2:
        assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
        assert a.is_contiguous(), 'Matrix A must be contiguous'
        assert b.is_contiguous(), 'Matrix B must be contiguous'
        M, K = a.shape
        K, N = b.shape
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_with_tensor_descriptors[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1))
    else:
        assert False, 'Input matrixs dimensions mismatch'
    return c


X_VALS = [[1, 1024 * i, 1024 * i, 1024 * i]
          for i in [1, 2, 4, 8]] + [[1, 1, 5120, 13824],  #
                                    [1, 4, 4096, 12288],  #
                                    [1, 512, 8192, 8192],  #
                                    [1, 512, 8192, 32768],  #
                                    [1, 512, 32768, 8192],  #
                                    [1, 1024, 16384, 8192],  #
                                    [1, 1024, 28672, 8192],  #
                                    [1, 3072, 4096, 3072],  # FIXME: Remove this case when gemm_streamk_benchmark works
                                    [1, 4096, 16384, 8192],  #
                                    [1, 8192, 16384, 1024],  #
                                    [1, 8192, 16384, 4096],  #
                                    [1, 16384, 1024, 8192],  #
                                    [1, 16384, 4096, 8192],  #
                                    [1, 16384, 8192, 1024],  #
                                    [1, 16384, 8192, 4096],  #
                                    [4, 32768, 128, 4096],  #
                                    [4, 32768, 4096, 128],  #
                                    [32, 4096, 4096, 128],  #
                                    [4096, 8, 128, 16384],  #
                                    [4096, 8, 16384, 128]]

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    # x_val: (B, M, K, N)
    B, M, K, N = x_val
    # a: (B, M, K) bfloat16
    # b: (B, K, N) bfloat16
    # c: (B, M, N) float32
    # pytorch reference: (B, M, N) float32
    required_memory = B * M * K * 2 + B * K * N * 2 + 2 * B * M * N * 4
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]


# Benchmark Performance
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'M', 'K', 'N'],
        # different possible values for `x_name`
        x_vals=X_VALS,
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['triton'],
        # label name for the lines
        line_names=['Triton'],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='matmul-performance-preop-exp',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, M, N, K, provider):
    # Some configs increase performance with warmup as a step function, but some slowly decrease with saturation. Performance is best at 200-400ms range, but we want stable, not just best
    # This warmup improves performance on BMG
    n_warmup = 800
    # We keep old warmup for now because longer warmup make perfomance on PVC worse
    n_warmup = 10

    if B == 1:
        a = torch.rand((M, K), device='xpu', dtype=torch.bfloat16)
        b = torch.rand((K, N), device='xpu', dtype=torch.bfloat16)
    else:
        a = torch.rand((B, M, K), device='xpu', dtype=torch.bfloat16)
        b = torch.rand((B, K, N), device='xpu', dtype=torch.bfloat16)

    quantiles = [0.5, 0.0, 1.0]

    if provider == 'triton':
        assert len(a.shape) == len(b.shape), 'Incompatible sizes'
        if len(a.shape) == 3:
            c = torch.empty((B, M, N), device='xpu', dtype=torch.float32)
        else:
            assert len(a.shape) == 2, 'Expecting shape of length 2'
            c = torch.empty((M, N), device='xpu', dtype=torch.float32)
        triton_fn = lambda: matmul(a, b, c)
        torch_fn = lambda: torch.matmul(torch.exp(a), b).to(torch.float32)
        rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
        benchmark_suit.assert_close(triton_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='triton to torch')
        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, n_warmup=n_warmup, n_repeat=10,
                                                                 quantiles=quantiles, time_warmup=False)
    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * B * M * N * K * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: B * (2 * (M * K + K * N) + 4.0 * (M * N)) * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
