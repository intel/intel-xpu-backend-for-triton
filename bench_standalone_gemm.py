"""
Standalone Triton GEMM kernel reproducer from issue #6012.
Dumps TTGIR, LLIR, and SPIR-V for analysis.
"""
import torch
import triton
import triton.language as tl
import os

# Enable IR dumps
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_issue6012"

@triton.jit
def triton_gemm_addmm(
    in_ptr0,  # bias
    arg_A,    # input matrix A
    arg_B,    # weight matrix B (transposed)
    out_ptr0, # output
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    ACC_TYPE = tl.float32
    INDEX_DTYPE = tl.int32

    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + K * idx_m
        a = tl.load(arg_A + xindex)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + N * idx_m
        b = tl.load(arg_B + (tl.broadcast_to(idx_m + K * idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape))

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # epilogue: add bias
    xindex = idx_n + N * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def run_benchmark():
    M, N, K = 10000, 64, 64

    bias = torch.randn(1, N, dtype=torch.float16, device="xpu")
    A = torch.randn(M, K, dtype=torch.float16, device="xpu")
    B = torch.randn(K, N, dtype=torch.float16, device="xpu")
    out = torch.empty(M, N, dtype=torch.float32, device="xpu")

    # Config matching the issue: BLOCK_M=32, BLOCK_N=32, BLOCK_K=64 (best autotuned)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    GROUP_M = 8

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n,)

    print(f"GEMM: M={M}, N={N}, K={K}")
    print(f"Config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, GROUP_M={GROUP_M}")
    print(f"Grid: {grid} ({grid_m}x{grid_n})")
    print()

    # Warmup
    for _ in range(5):
        triton_gemm_addmm[grid](
            bias, A, B, out,
            M, N, K,
            K, 1, 1, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=4, num_stages=2,
        )
    torch.xpu.synchronize()

    # Benchmark
    import time
    num_iters = 200
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(num_iters):
        triton_gemm_addmm[grid](
            bias, A, B, out,
            M, N, K,
            K, 1, 1, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=4, num_stages=2,
        )
    torch.xpu.synchronize()
    end = time.perf_counter_ns()

    avg_us = (end - start) / num_iters / 1000
    flops = 2 * M * N * K
    gflops = flops / ((end - start) / num_iters)
    print(f"Triton GEMM avg: {avg_us:.1f} us")
    print(f"GFLOPS: {gflops:.2f}")

    # Also test the 64x64x32 config from the issue
    BLOCK_M2, BLOCK_N2, BLOCK_K2 = 64, 64, 32
    grid_m2 = (M + BLOCK_M2 - 1) // BLOCK_M2
    grid_n2 = (N + BLOCK_N2 - 1) // BLOCK_N2
    grid2 = (grid_m2 * grid_n2,)

    out2 = torch.empty(M, N, dtype=torch.float32, device="xpu")
    for _ in range(5):
        triton_gemm_addmm[grid2](
            bias, A, B, out2,
            M, N, K,
            K, 1, 1, K,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2, GROUP_M=GROUP_M,
            num_warps=4, num_stages=2,
        )
    torch.xpu.synchronize()

    torch.xpu.synchronize()
    start2 = time.perf_counter_ns()
    for _ in range(num_iters):
        triton_gemm_addmm[grid2](
            bias, A, B, out2,
            M, N, K,
            K, 1, 1, K,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2, GROUP_M=GROUP_M,
            num_warps=4, num_stages=2,
        )
    torch.xpu.synchronize()
    end2 = time.perf_counter_ns()

    avg_us2 = (end2 - start2) / num_iters / 1000
    gflops2 = flops / ((end2 - start2) / num_iters)
    print(f"\nTriton GEMM (64x64x32 config) avg: {avg_us2:.1f} us")
    print(f"GFLOPS: {gflops2:.2f}")

    # oneDNN reference
    torch.xpu.synchronize()
    start3 = time.perf_counter_ns()
    for _ in range(num_iters):
        y = torch.addmm(bias, A, B)
    torch.xpu.synchronize()
    end3 = time.perf_counter_ns()

    avg_us3 = (end3 - start3) / num_iters / 1000
    gflops3 = flops / ((end3 - start3) / num_iters)
    print(f"\noneDNN GEMM avg: {avg_us3:.1f} us")
    print(f"GFLOPS: {gflops3:.2f}")
    print(f"\nTriton/oneDNN ratio (32x32x64): {avg_us / avg_us3:.2f}x")
    print(f"Triton/oneDNN ratio (64x64x32): {avg_us2 / avg_us3:.2f}x")


if __name__ == "__main__":
    run_benchmark()
