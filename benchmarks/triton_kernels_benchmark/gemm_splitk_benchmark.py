import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4, 'SPLIT_K': 4, "threads_per_warp": 16},
            num_stages=4, num_warps=32),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _kernel(A, B, C, M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            acc_dtype: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,  #
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

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        acc += tl.dot(a, b, out_dtype=acc_dtype)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    @staticmethod
    def _call(a, b, acc_dtype, output_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        # allocates output
        if (output_dtype is None):
            output_dtype = torch.float32

        c = torch.empty((M, N), device=device, dtype=output_dtype)

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
        }

        if acc_dtype is None:
            acc_dtype = torch.float32
        else:
            assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
            assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
            assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

        def to_tl_type(ty):
            return getattr(tl, str(ty).split(".")[-1])

        acc_dtype = to_tl_type(acc_dtype)
        output_dtype = to_tl_type(output_dtype)

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](
            a, b, c, M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            acc_dtype=acc_dtype)
        return c

    @staticmethod
    def forward(ctx, a, b, acc_dtype=None, output_dtype=None):
        return _matmul._call(a, b, acc_dtype=acc_dtype, output_dtype=output_dtype)


matmul = _matmul.apply

# %%
# Unit Test
# ---------
#
# Still we can test our matrix multiplication with block pointers against a native torch implementation (i.e., OneDNN).
torch.manual_seed(0)
for dtype in [torch.float16]:
    a = torch.randn((512, 512), device='xpu', dtype=dtype)
    b = torch.randn((512, 512), device='xpu', dtype=dtype)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b).to(torch.float32)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")

    # Note: the torch.matmul and Triton implementations uses different
    # algorithms so we need to adjust tolerance.
    rtol = 1e-3
    if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")


# Benchmark Performance
@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'K', 'N'],
        x_vals=[
            [512, 32768, 8192],
            [3072, 4096, 3072],
            [4096, 4096, 4096],
            [1024, 28672, 8192],
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['onednn', 'triton'],
        # label name for the lines
        line_names=["onednn", "Triton"],
        # line styles
        # styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    torch.manual_seed(0)
    a = torch.rand((M, K), device='xpu', dtype=torch.bfloat16)
    b = torch.rand((K, N), device='xpu', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    # calculate tflops for oneDNN kernel
    def calculate_tflops(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    if provider == 'onednn':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=100, quantiles=quantiles,
                                                     fast_flush=False)
        print(f"oneDNN Peak TFlops {calculate_tflops(min_ms)}")
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), rep=100, quantiles=quantiles,
                                                     fast_flush=False)

    return calculate_tflops(ms), calculate_tflops(min_ms), calculate_tflops(max_ms)


benchmark.run(show_plots=True, print_data=True)
