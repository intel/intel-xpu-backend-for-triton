"""
Gemm benchmark (tensor descriptor)
============================

This benchmark uses tensor descriptors to implement a GEMM kernel.
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
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator += tl.dot(a, b)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)

    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


# pylint: disable=unused-argument
@triton.autotune(
    configs=gemm_benchmark.get_matmul_batched_autotune_configs(),
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
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator += tl.dot(a, b)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)

    offset_c = bid.to(tl.int64) * stride_cz
    c_desc = tl.make_tensor_descriptor(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))

    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


def get_benchmark(
    providers_filter: Optional[List[str]] = None,
    transpose_a=False,
    transpose_b=False,
):
    return gemm_benchmark.get_benchmark(
        providers_filter=providers_filter,
        matmul_kernel=matmul_kernel_with_tensor_descriptors,
        matmul_kernel_batched=matmul_kernel_with_tensor_descriptors_batched,
        plot_name='matmul-tensor-desc-performance',
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


if __name__ == '__main__':
    _benchmark = get_benchmark(
        transpose_a=(os.getenv('TRANSPOSE_A', '0') == '1'),
        transpose_b=(os.getenv('TRANSPOSE_B', '0') == '1'),
    )
    _benchmark.run(show_plots=False, print_data=True)
