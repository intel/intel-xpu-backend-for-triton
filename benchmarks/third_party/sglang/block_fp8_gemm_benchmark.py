"""
Block FP8 Gemm benchmark
============================
This benchmark is come from SGLang kernels.
https://github.com/sgl-project/sglang/blob/07f944631e747d7489fde1f11de93e503afa90ba/python/sglang/srt/layers/quantization/fp8_kernel.py#L375
"""

from typing import List

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suit

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N, )
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Default config
    # Block-wise quant: BLOCK_SIZE_K must be divisable by block_size[1]
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_size[0],
        "BLOCK_SIZE_K": block_size[1],
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    kernel = _w8a8_block_fp8_matmul

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


# For test
def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """This function performs matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def has_enough_memory(x_val):
    # x_val: (M, N, K)
    M, N, K = x_val
    # a: (M, K) float8_e4m3
    # b: (N, K) float8_e4m3
    # c: (M, N) bfloat16
    # pytorch reference: (M, N) float32
    required_memory = M * K * 1 + N * K * 1 + M * N * 2 * 2
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [[1024 * i, 1024 * i, 1024 * i] for i in [1, 2, 4, 8]] + [
    [1, 13824, 5120],
    [4, 12288, 4096],
    [512, 8192, 8192],
    [512, 8192, 32768],
    [512, 32768, 8192],
    [1024, 8192, 16384],
    [1024, 8192, 28672],
    [3072, 3072, 4096],
    [4096, 8192, 16384],
    [8192, 1024, 16384],
    [8192, 4096, 16384],
    [16384, 1024, 8192],
    [16384, 4096, 8192],
    [16384, 8192, 1024],
    [16384, 8192, 4096],
    [32768, 128, 4096],
    [32768, 4096, 128],
    [4096, 128, 4096],
    [8, 128, 16384],
    [8, 16384, 128],
]

X_VALS = [x_val for x_val in X_VALS if has_enough_memory(x_val)]


# Benchmark Performance
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["M", "N", "K"],
        # different possible values for `x_name`
        x_vals=X_VALS,
        line_arg="provider",
        # argument name whose value corresponds to a different line in the plot
        line_vals=["triton"],
        # label name for the lines
        line_names=["Triton"],
        # line styles
        ylabel=["GB/s", "TFlops"],  # label name for the y-axis
        plot_name="sglang-fp8-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    torch.manual_seed(0)

    block_size = [128, 128]
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="xpu") - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="xpu") - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="xpu") * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="xpu") * factor_for_scale

    quantiles = [0.5, 0.0, 1.0]

    if provider == "triton":
        triton_fn = lambda: w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size)
        torch_fn = lambda: native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size)
        benchmark_suit.assert_close(triton_fn, torch_fn, atol=3e-4, rtol=1e-2, err_msg="triton to torch")
        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10,
                                                                 quantiles=quantiles)

    else:
        raise NotImplementedError(f"Unsupported provider {provider}")

    tflops = lambda ms: 2 * M * N * K * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: (M * K + K * N) + 2.0 * (M * N) * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)
