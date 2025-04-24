"""
Gemm benchmark (tensor of pointer)
============================

This benchmark is come from the Triton tutorial 03-matrix-multiplication.py (commit: 3f4fdd1)
To compare the performance to XeTLA kernel.

"""
from typing import List, Optional

import os

import triton
import triton.language as tl

from triton_kernels_benchmark import gemm_benchmark


@triton.autotune(
    configs=gemm_benchmark.get_matmul_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# pylint: disable=unused-argument
@triton.autotune(
    configs=gemm_benchmark.get_matmul_batched_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_batched(
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz
    a_ptrs = a_ptr + offset_a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offset_b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_c = bid.to(tl.int64) * stride_cz
    c_ptrs = c_ptr + offset_c + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def get_benchmark(
    providers_filter: Optional[List[str]] = None,
    transpose_a=False,
    transpose_b=False,
):
    return gemm_benchmark.get_benchmark(
        providers_filter=providers_filter,
        matmul_kernel=matmul_kernel,
        matmul_kernel_batched=matmul_kernel_batched,
        plot_name='matmul-tensor-of-ptr-performance',
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


if __name__ == '__main__':
    _benchmark = get_benchmark(
        transpose_a=(os.getenv('TRANSPOSE_A', '0') == '1'),
        transpose_b=(os.getenv('TRANSPOSE_B', '0') == '1'),
    )
    _benchmark.run(show_plots=False, print_data=True)
