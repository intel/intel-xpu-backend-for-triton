"""
Fused GEMM with SwiGLU benchmark
=================================

This benchmark measures the performance of a kernel that fuses two parallel GEMM
operations with a SwiGLU post-operation:
    y = silu(x @ w_g + b_g) * (x @ w_fc + b_fc)

This pattern is common in LLM architectures (e.g., Llama, Mistral) as the
feed-forward network's SwiGLU activation layer.

Multiple kernel variants are benchmarked to compare optimizations:
  - baseline: original kernel with fast_dividef/fast_expf for SiLU
  - bias_init: initialize accumulators with bias (saves 2 post-loop vec adds)
  - rounded_div: use tl.math.div_rn (SPV_INTEL_rounded_divide_sqrt) for SiLU
  - sigmoid: use tl.sigmoid(x) * x for SiLU
  - large_k: baseline with BLOCK_SIZE_K=128 added to autotune space

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


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

def get_autotune_configs(block_k_values=None) -> list[triton.Config]:
    if block_k_values is None:
        block_k_values = [32, 64]
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
        for BK in block_k_values
        for G in [4, 8, 16]
        for s in [2, 3, 4]
        for w in [8, 16, 32]
    ]


# ---------------------------------------------------------------------------
# Variant 1: Baseline (original kernel)
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_baseline(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

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
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Variant 2: Bias initialization in accumulators (saves post-loop addition)
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_bias_init(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

    # Load bias and broadcast into accumulator shape — avoids post-loop addition
    offset_n = off_n + tl.arange(0, BLOCK_SIZE_N)
    b_g = tl.load(b_g_ptr + offset_n, mask=offset_n < N, other=0.0).to(tl.float32)
    b_fc = tl.load(b_fc_ptr + offset_n, mask=offset_n < N, other=0.0).to(tl.float32)

    acc_g = tl.broadcast_to(b_g[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    acc_fc = tl.broadcast_to(b_fc[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))

    for k in range(0, K, BLOCK_SIZE_K):
        x = desc_x.load([off_m, k])
        w_g = desc_wg.load([k, off_n])
        w_fc = desc_wfc.load([k, off_n])
        acc_g = tl.dot(x, w_g, acc_g)
        acc_fc = tl.dot(x, w_fc, acc_fc)

    silu_g = libdevice.fast_dividef(acc_g, 1.0 + libdevice.fast_expf(-acc_g))
    y = (silu_g * acc_fc).to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Variant 3: Rounded divide (SPV_INTEL_rounded_divide_sqrt) for SiLU
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_rounded_div(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

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

    # Use rounded divide (SPV_INTEL_rounded_divide_sqrt) instead of fast_dividef
    silu_g = tl.math.div_rn(acc_g, 1.0 + libdevice.fast_expf(-acc_g))
    y = (silu_g * acc_fc).to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Variant 4: sigmoid-based SiLU — silu(x) = x * sigmoid(x)
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_sigmoid(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

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

    # SiLU via sigmoid: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_g = acc_g * tl.sigmoid(acc_g)
    y = (silu_g * acc_fc).to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Variant 5: Larger BLOCK_SIZE_K (adds K=128 to autotune space)
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(block_k_values=[32, 64, 128]), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_large_k(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

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
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Variant 6: Combined best — bias_init + sigmoid + large_k
# ---------------------------------------------------------------------------

@triton.autotune(configs=get_autotune_configs(block_k_values=[32, 64, 128]), key=['M', 'N', 'K'], restore_value=['y_ptr'])
@triton.jit
def fused_gemm_combined(
    x_ptr, w_g_ptr, w_fc_ptr, b_g_ptr, b_fc_ptr, y_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

    offset_n = off_n + tl.arange(0, BLOCK_SIZE_N)
    b_g = tl.load(b_g_ptr + offset_n, mask=offset_n < N, other=0.0).to(tl.float32)
    b_fc = tl.load(b_fc_ptr + offset_n, mask=offset_n < N, other=0.0).to(tl.float32)

    acc_g = tl.broadcast_to(b_g[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    acc_fc = tl.broadcast_to(b_fc[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))

    for k in range(0, K, BLOCK_SIZE_K):
        x = desc_x.load([off_m, k])
        w_g = desc_wg.load([k, off_n])
        w_fc = desc_wfc.load([k, off_n])
        acc_g = tl.dot(x, w_g, acc_g)
        acc_fc = tl.dot(x, w_fc, acc_fc)

    silu_g = acc_g * tl.sigmoid(acc_g)
    y = (silu_g * acc_fc).to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_y.store([off_m, off_n], y)


# ---------------------------------------------------------------------------
# Dispatch map: provider name -> kernel function
# ---------------------------------------------------------------------------

KERNEL_VARIANTS = {
    'baseline': fused_gemm_baseline,
    'bias_init': fused_gemm_bias_init,
    'rounded_div': fused_gemm_rounded_div,
    'sigmoid': fused_gemm_sigmoid,
    'large_k': fused_gemm_large_k,
    'combined': fused_gemm_combined,
}


# Shapes from PR #7314 (customer model + DeepSeek-R1 style).
X_VALS = [
    [220000, 8192, 512],  # shape from customer model
    [1024, 8192, 7168],
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


def fused_gemm_swiglu(kernel_fn, x, w_g, w_fc, b_g, b_fc, M, N, K):
    y = torch.empty((M, N), device='xpu', dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel_fn[grid](x, w_g, w_fc, b_g, b_fc, y, M, N, K)
    return y


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]

VARIANT_NAMES = list(KERNEL_VARIANTS.keys())


@benchmark_suite.perf_report(
    benchmark_suite.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=X_VALS,
        line_arg='provider',
        line_vals=VARIANT_NAMES,
        line_names=VARIANT_NAMES,
        styles=[
            ('green', '-'),       # baseline
            ('blue', '-'),        # bias_init
            ('red', '-'),         # rounded_div
            ('orange', '-'),      # sigmoid
            ('purple', '-'),      # large_k
            ('black', '--'),      # combined
        ],
        ylabel=['GB/s', 'TFlops'],
        plot_name='fused-gemm-swiglu-variants',
        args={},
    ))
def benchmark(M, N, K, provider):
    do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])

    torch.xpu.empty_cache()
    torch.manual_seed(0)
    x = torch.empty((M, K), device='xpu', dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
    w_g = torch.empty((K, N), device='xpu', dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
    w_fc = torch.empty((K, N), device='xpu', dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
    b_g = torch.zeros((N, ), device='xpu', dtype=torch.bfloat16)
    b_fc = torch.zeros((N, ), device='xpu', dtype=torch.bfloat16)

    kernel_fn = KERNEL_VARIANTS[provider]
    triton_fn = lambda: fused_gemm_swiglu(kernel_fn, x, w_g, w_fc, b_g, b_fc, M, N, K)
    _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)

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
