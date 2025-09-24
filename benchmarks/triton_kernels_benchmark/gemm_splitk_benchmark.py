import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark import xetla_kernel


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4, 'SPLIT_K': 4, 'grf_mode': 'large'},
                      num_stages=4, num_warps=32),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _kernel(A, B, C,  #
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
            stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
            stride_cm: tl.constexpr, stride_cn: tl.constexpr,  #
            acc_dtype: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr  #
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    a_block_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, pid_z * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(pid_z * BLOCK_K, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for _ in range(0, K, BLOCK_K * SPLIT_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b, out_dtype=acc_dtype)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K * SPLIT_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K * SPLIT_K, 0))
    acc = acc.to(C.dtype.element_ty)
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        c_block_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                        order=(1, 0))
        tl.store(c_block_ptr, acc, boundary_check=(0, 1))
    else:
        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(C, acc, mask=mask, sem='relaxed')


class _matmul(torch.autograd.Function):
    kernel = _kernel

    @staticmethod
    def _call(a, b, c, acc_dtype):
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], 'incompatible dimensions'
        M, K = a.shape
        _, N = b.shape

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
        }

        if acc_dtype is None:
            acc_dtype = torch.float32
        else:
            assert isinstance(acc_dtype, torch.dtype), 'acc_dtype must be a torch.dtype'
            assert acc_dtype in supported_acc_dtypes[a.dtype], 'acc_dtype not compatible with the type of a'
            assert acc_dtype in supported_acc_dtypes[b.dtype], 'acc_dtype not compatible with the type of b'

        def to_tl_type(ty):
            return getattr(tl, str(ty).rsplit('.', maxsplit=1)[-1])

        acc_dtype = to_tl_type(acc_dtype)

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](
            a, b, c, M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            acc_dtype=acc_dtype)
        return c

    # pylint: disable=unused-argument
    @staticmethod
    def forward(ctx, a, b, c, acc_dtype=None):
        return _matmul._call(a, b, c, acc_dtype=acc_dtype)


matmul = _matmul.apply


# Benchmark Performance
@benchmark_suite.perf_report(
    benchmark_suite.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'K', 'N'],
        x_vals=[
            [512, 32768, 8192],
            [1024, 28672, 8192],
            [3072, 4096, 3072],
            [4096, 4096, 4096],
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['triton', 'xetla'],
        # label name for the lines
        line_names=['Triton', 'XeTLA'],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='matmul-splitk-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    # Maximum across onednn=10, triton=100, xetla=300
    n_warmup = 300
    torch.manual_seed(0)
    a = torch.rand((M, K), device='xpu', dtype=torch.bfloat16)
    b = torch.rand((K, N), device='xpu', dtype=torch.bfloat16)
    quantiles = [0.5, 0.0, 1.0]

    if provider == 'onednn':
        _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(lambda: torch.matmul(a, b), n_warmup=n_warmup,
                                                                  n_repeat=10, quantiles=quantiles)
    elif provider == 'triton':
        c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
        triton_fn = lambda: matmul(a, b, c)
        torch_fn = lambda: torch.matmul(a, b).to(torch.float32)
        rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
        benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='triton to torch')
        _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(triton_fn, n_warmup=n_warmup, n_repeat=10,
                                                                  quantiles=quantiles)
    elif provider == 'xetla':
        c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
        acc = torch.zeros((M, N), device='xpu', dtype=torch.float32)
        cnt = torch.zeros((M, N), device='xpu', dtype=torch.int32)

        name = f'gemm_splitk_shape_{M}_{K}_{N}'
        func = getattr(xetla_kernel, name)
        xetla_fn = lambda: func(a, b, c, acc, cnt)
        torch_fn = lambda: torch.matmul(a, b).to(torch.float32)

        # benchmark_suite.assert_close(xetla_fn, torch_fn, atol=1e-4, rtol=1.0, err_msg='xetla to torch')
        _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(xetla_fn, n_warmup=n_warmup, n_repeat=100,
                                                                  quantiles=quantiles)
    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda mean: 2 * M * N * K * (1e-12) / (mean * 1e-3)
    gbps = lambda mean: 2 * (M * K + K * N) + 4.0 * (M * N) * (1e-9) / (mean * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
