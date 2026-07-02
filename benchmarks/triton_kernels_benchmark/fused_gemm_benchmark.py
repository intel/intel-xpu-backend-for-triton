"""
Fused GEMM with SwiGLU benchmark
=================================

This benchmark measures the performance of a kernel that fuses two parallel GEMM
operations with a SwiGLU post-operation:
    y = silu(x @ w_g + b_g) * (x @ w_fc + b_fc)

This pattern is common in LLM architectures (e.g., Llama, Mistral) as the
feed-forward network's SwiGLU activation layer.

"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import triton_kernels_benchmark as benchmark_suite


def native_torch_fused_gemm(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(x @ w_g + b_g)
    fc = x @ w_fc + b_fc
    return gate * fc


def get_fused_gemm_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM,
                'BLOCK_SIZE_N': BN,
                'BLOCK_SIZE_K': BK,
                'GROUP_SIZE_M': G,
                'grf_mode': '256',
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128, 256]
        for BN in [64, 128]
        for BK in [32, 64]
        for G in [4, 8, 16]
        for s in [2, 3, 4]
        for w in [8, 16, 32]
    ]


@triton.autotune(
    configs=get_fused_gemm_autotune_configs(),
    key=['M', 'N', 'K'],
    restore_value=['y_ptr'],
)
@triton.jit
def fused_gemm_swiglu_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = y_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    desc_x = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )

    offset_n = off_n + tl.arange(0, BLOCK_SIZE_N)
    b_g = tl.load(b_g_ptr + offset_n, mask=offset_n < N, other=0.0)
    b_fc = tl.load(b_fc_ptr + offset_n, mask=offset_n < N, other=0.0)

    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = desc_x.load([off_m, k])
        w_g = desc_wg.load([k, off_n])
        w_fc = desc_wfc.load([k, off_n])
        acc_g = tl.dot(x, w_g, acc_g)
        acc_fc = tl.dot(x, w_fc, acc_fc)

    acc_g += b_g[None, :]
    acc_fc += b_fc[None, :]

    silu_g = libdevice.fast_dividef(acc_g, 1.0 + libdevice.fast_expf(-acc_g))
    y = (silu_g * acc_fc).to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    desc_y.store([off_m, off_n], y)


# Representative shapes for LLM feed-forward SwiGLU layers.
# Each entry is [M, N, K] where x is (M, K), w_g/w_fc are (K, N), and y is (M, N).
X_VALS = [
    [220000, 8192, 512],  # shape from customer model
    [1024, 8192, 7168],
    [1024, 5504, 4096],  # Llama-2 7B
    [2048, 5504, 4096],
    [4096, 5504, 4096],
    [8192, 5504, 4096],
    [1024, 14336, 4096],  # Llama-3 8B
    [2048, 14336, 4096],
    [4096, 14336, 4096],
    [8192, 14336, 4096],
    # TODO: Fix the bug of kernel implementation on shape from DeepSeek-R1.
    # [1024, 8192, 7168],  # DeepSeek-R1 style
    # [4096, 8192, 7168],
    # [8192, 8192, 7168],
]

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    # x_val: (M, N, K)
    M, N, K = x_val
    # x: (M, K) bfloat16, w_g: (K, N) bfloat16, w_fc: (K, N) bfloat16
    # b_g, b_fc: (N,) bfloat16, y: (M, N) bfloat16, torch reference: (M, N) bfloat16
    required_memory = (M * K + 2 * K * N + 2 * N + 2 * M * N) * 2
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


def is_enough_memory_for_verification(M, N, K):
    """Check if there is enough device memory to run accuracy verification.

    assert_close requires computing the PyTorch reference, which allocates several
    large (M, N) intermediate tensors (float32 + bfloat16). We conservatively
    estimate 18 * M * N bytes of additional memory on top of the benchmark inputs.
    """
    # Existing benchmark tensors: x (M*K), w_g (K*N), w_fc (K*N), b_g (N), b_fc (N), y (M*N)
    input_memory = (M * K + 2 * K * N + 2 * N + M * N) * 2  # bfloat16 bytes
    # PyTorch reference intermediates: x@w_g (float32 MN), silu (float32 MN),
    # x@w_fc (float32 MN), gate (bf16 MN), fc (bf16 MN), ref output (bf16 MN)
    # ≈ 3*4*MN + 3*2*MN = 18*MN bytes (conservative with safety margin)
    verify_memory = 18 * M * N
    return (input_memory + verify_memory) < DEVICE_TOTAL_MEMORY


def fused_gemm_swiglu(x, w_g, w_fc, b_g, b_fc, M, N, K):
    y = torch.empty((M, N), device='xpu', dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_gemm_swiglu_kernel[grid](x, w_g, w_fc, b_g, b_fc, y, M, N, K)
    return y


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]


@benchmark_suite.perf_report(
    benchmark_suite.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        # different possible values for `x_name`
        x_vals=X_VALS,
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg`
        line_vals=['triton'],
        # label name for the lines
        line_names=['Triton'],
        # line styles
        styles=[('green', '-')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='fused-gemm-swiglu-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])

    torch.xpu.empty_cache()
    torch.manual_seed(0)
    if M == 220000 and N == 8192 and K == 512:
        x = torch.empty((M, K), device='xpu', dtype=torch.float32).uniform_(-0.25, -0.25).to(torch.bfloat16)
        w_g = torch.empty((K, N), device='xpu', dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
        w_fc = torch.empty((K, N), device='xpu', dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
    else:
        x = torch.rand((M, K), device='xpu', dtype=torch.bfloat16)
        w_g = torch.rand((K, N), device='xpu', dtype=torch.bfloat16)
        w_fc = torch.rand((K, N), device='xpu', dtype=torch.bfloat16)
    b_g = torch.zeros((N, ), device='xpu', dtype=torch.bfloat16)
    b_fc = torch.zeros((N, ), device='xpu', dtype=torch.bfloat16)

    if provider == 'triton':
        triton_fn = lambda: fused_gemm_swiglu(x, w_g, w_fc, b_g, b_fc, M, N, K)
        torch_fn = lambda: native_torch_fused_gemm(x, w_g, w_fc, b_g, b_fc)
        if is_enough_memory_for_verification(M, N, K):
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-2, rtol=1e-2, err_msg='triton to torch')
        else:
            print(f"Skipping accuracy verification for shape ({M}, {N}, {K}): "
                  f"insufficient device memory to compute PyTorch reference")
        _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    # Two parallel GEMMs (w_g and w_fc), each with 2 * M * N * K multiply-add FLOPs
    num_gemms = 2
    tflops = lambda ms: num_gemms * 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # Memory: x (M*K), w_g (K*N), w_fc (K*N), b_g (N), b_fc (N), y (M*N) -- all bfloat16 (2 bytes)
    gbps = lambda ms: (M * K + 2 * K * N + 2 * N + M * N) * 2 * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


# pylint: disable=unused-argument
def get_benchmark(providers_filter=None):
    return benchmark


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
