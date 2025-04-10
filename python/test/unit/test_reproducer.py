import itertools

import torch
import triton
import triton.language as tl



@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    s1_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_s1m,
    stride_s1n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.int32,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
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
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)  # , allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N * idx_m)
    tmp0 = tl.load(
        s1_ptr + (tl.broadcast_to(idx_m, mask.shape)),
        mask,
        eviction_policy="evict_last",
    )
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)





def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, K = a.shape
    K, N = b.shape
    print("a.sizes(): ", a.size(), "a.strides(): ", a.stride(), "a.dtype: ", a.dtype)
    print("b.sizes(): ", b.size(), "b.strides(): ", b.stride(), "b.dtype: ", b.dtype)
    print("c.sizes(): ", c.size(), "c.strides(): ", c.stride(), "c.dtype: ", c.dtype)
    print("scales1.sizes(): ", scales1.size(), "scales1.strides(): ", scales1.stride(), "scales1.dtype", scales1.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    scaled_matmul_kernel_with_block_pointers[grid](
        a,
        b,
        c,
        scales1,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),
        scales1.stride(0),
        scales1.stride(1),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        EVEN_K=(K % 2 == 0),
        **config.kwargs,
    )
    return c

import torch

def test_reproducer():
    # Generate random input tensors with the specified properties
    a = torch.randint(-128, 127, (1, 1024), dtype=torch.int8, device="xpu").as_strided((1, 1024), (1024, 1))
    b = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8, device="xpu").as_strided((1024, 1024), (1, 1024))
    c = torch.empty((1, 1024), dtype=torch.bfloat16, device="xpu").as_strided((1, 1024), (1024, 1))
    scales1 = torch.empty((1, 1024), dtype=torch.bfloat16, device="xpu").as_strided((1, 1024), (1, 0))
    config =    triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=1,
            num_warps=2,
            num_ctas=1,
            maxnreg=None
        )

    print(int_scaled_matmul_kernel(a, b, scales1, c, config))
