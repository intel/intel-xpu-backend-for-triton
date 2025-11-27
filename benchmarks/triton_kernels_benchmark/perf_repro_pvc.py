from typing import Callable, List, Optional
import os

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language.intel import IntelDPASLayout
from triton.tools.tensor_descriptor import TensorDescriptor


import triton_kernels_benchmark as benchmark_suite
# from triton_kernels_benchmark import xetla_kernel
# from triton_kernels_benchmark import cutlass_kernel

from pudb import set_trace



@triton.jit
def matmul_kernel_with_block_pointers(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(tl.float32)

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@gluon.jit
def gluon_matmul_kernel_dpas(
    a_ptr, b_ptr, c_ptr,
    M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,
    stride_am: ttgl.constexpr, stride_ak: ttgl.constexpr,
    stride_bk: ttgl.constexpr, stride_bn: ttgl.constexpr,
    stride_cm: ttgl.constexpr, stride_cn: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr
):
    # Define DPAS layout - adjust parameters based on your matrix sizes
    layout: ttgl.constexpr = IntelDPASLayout(
        repeatCount=8,
        systolic_depth=8,
        execution_size=16,
        ops_per_chan=2,  # For bf16/f16
        warps_per_cta=[8, 4],  # Adjust based on num_warps
        rep_cluster=[4, 2],
        threads_per_warp=16  # Or 32 with env var
    )

    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=layout, operand_index=0, k_width=1
    )
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=layout, operand_index=1, k_width=2
    )

    # Program ID and block calculation (same as standard Triton)
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ttgl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize accumulator with DPAS layout
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=layout)

    # Manual offset calculation for A (replaces make_block_ptr)
    offs_am = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout)))[:, None]
    offs_ak = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, layout))[None, :]
    offs_a = offs_am * stride_am + offs_ak * stride_ak

    # Manual offset calculation for B
    offs_bk = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, layout))[:, None]
    offs_bn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout)))[None, :]
    offs_b = offs_bk * stride_bk + offs_bn * stride_bn

    # K-dimension loop
    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        # Load A and convert to dot operand layout
        a = ttgl.load(a_ptr + offs_a)
        a = ttgl.convert_layout(a, lhs_layout)

        # Load B and convert to dot operand layout
        b = ttgl.load(b_ptr + offs_b)
        b = ttgl.convert_layout(b, rhs_layout)

        # Perform dot operation (uses DPAS instructions)
        accumulator = ttgl.xpu_dot_fma(a, b, accumulator)

        # Advance pointers (replaces tl.advance)
        offs_a += BLOCK_K * stride_ak
        offs_b += BLOCK_K * stride_bk

    # Store result
    offs_cm = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout)))[:, None]
    offs_cn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout)))[None, :]
    offs_c = offs_cm * stride_cm + offs_cn * stride_cn

    ttgl.store(c_ptr + offs_c, accumulator)


@gluon.jit
def gluon_matmul_kernel_dpas_tensor_desc(
    a_ptr, b_ptr, c_ptr,
    M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,
    stride_am: ttgl.constexpr, stride_ak: ttgl.constexpr,
    stride_bk: ttgl.constexpr, stride_bn: ttgl.constexpr,
    stride_cm: ttgl.constexpr, stride_cn: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr
):
    layout: ttgl.constexpr = IntelDPASLayout(
        repeatCount=8,
        systolic_depth=8,
        execution_size=16,
        ops_per_chan=2,
        warps_per_cta=[8, 4],
        rep_cluster=[4, 2],
        threads_per_warp=16
    )

    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=layout, operand_index=0, k_width=1
    )
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=layout, operand_index=1, k_width=2
    )

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ttgl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_desc = ttgl.intel.xpu.td.make_tensor_descriptor(
        a_ptr,
        (M, K),
        (stride_am, stride_ak),
        (BLOCK_M, BLOCK_K),
        lhs_layout
    )
    b_desc = ttgl.intel.xpu.td.make_tensor_descriptor(
        b_ptr,
        (K, N),
        (stride_bk, stride_bn),
        (BLOCK_K, BLOCK_N),
        rhs_layout
    )
    c_desc = ttgl.intel.xpu.td.make_tensor_descriptor(
        c_ptr,
        (M, N),
        (stride_cm, stride_cn),
        (BLOCK_M, BLOCK_N),
        layout
    )

    accumulator = ttgl.load_tensor_descriptor(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N])

    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        a = ttgl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_M, k * BLOCK_K])
        b = ttgl.load_tensor_descriptor(b_desc, [k * BLOCK_K, pid_n * BLOCK_N])

        accumulator = ttgl.xpu_dot_fma(a, b, accumulator)

    ttgl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)



def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, grf_mode, num_warps, num_ctas, num_stages, maxnreg):
    a_major, a_minor = -2, -1
    b_minor, b_major = -2, -1

    assert a.shape[a_minor] == b.shape[b_minor], 'Incompatible dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    assert b.is_contiguous(), 'Matrix B must be contiguous'

    M, N, K = a.shape[a_major], b.shape[b_major], a.shape[a_minor]

    # Check constraints.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_with_block_pointers[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(a_major), a.stride(a_minor),  #
        b.stride(b_minor), b.stride(b_major),  #
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M,
        grf_mode=grf_mode,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
        # maxnreg=maxnreg
    )
    return c



def gluon_matmul(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, grf_mode, num_warps, num_ctas, num_stages, maxnreg):
    """Wrapper for Gluon DPAS matmul kernel"""
    M, K = a.shape
    K, N = b.shape

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    print(f'{grid}')
    gluon_matmul_kernel_dpas[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        grf_mode=grf_mode,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
        # maxnreg=maxnreg
    )
    return c


def gluon_matmul_tensor_descriptors(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, grf_mode, num_warps, num_ctas, num_stages, maxnreg):
    """Wrapper for Gluon DPAS matmul kernel"""
    M, K = a.shape
    K, N = b.shape

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    print(f'{grid}')
    gluon_matmul_kernel_dpas_tensor_desc[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        grf_mode=grf_mode,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
        # maxnreg=maxnreg
    )
    return c


def get_shapes(B, M, N, K, transpose_a=False, transpose_b=False):
    a_shape = (M, K)
    if transpose_a:
        a_shape = (K, M)

    b_shape = (K, N)
    if transpose_b:
        b_shape = (N, K)

    if B != 1:
        a_shape = (B, *a_shape)
        b_shape = (B, *b_shape)
    return a_shape, b_shape



if __name__ == "__main__":
    B, M, N, K = [1, 1, 1024, 4096]
    print(f'# {B}-{M}-{N}-{K}')
    # Maximum across onednn=600, triton=800, xetla=10, cutlass=600
    #do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])

    a_shape, b_shape = get_shapes(B, M, N, K)
    print(f'{a_shape=}')
    torch.xpu.synchronize()
    print(f'{b_shape=}')

    torch.manual_seed(0)
    a = torch.rand(a_shape, device='xpu', dtype=torch.bfloat16)
    b = torch.rand(b_shape, device='xpu', dtype=torch.bfloat16)


    if len(a.shape) != len(b.shape):
        raise AssertionError(f'Incompatible sizes {len(a.shape)} and {len(b.shape)}', )
    if len(a.shape) == 3:
        c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
    elif len(a.shape) == 2:
        c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
    else:
        raise AssertionError(f'Unexpected shape of length {len(a.shape)}')

    # Fixed block sizes for Gluon (no autotuning)
    # PVC
    # best config selected: BLOCK_SIZE_M: 256, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 4, grf_mode: 256, num_warps: 32, num_ctas: 1, num_stages: 4, maxnreg: None

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128 , 32 # Autotuned for triton.jit & PVC
    GROUP_SIZE_M = 4
    GRF_MODE = '256'
    NUM_WARPS = 32
    NUM_CTAS = 1
    NUM_STAGES = 4
    MAXNREG = None


    run_gluon = True

    if run_gluon:
        ## Gluon
        gluon_fn = lambda: gluon_matmul_tensor_descriptors(
            a, b, c,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            grf_mode=GRF_MODE,
            num_warps=NUM_WARPS,
            num_ctas=NUM_CTAS,
            num_stages=NUM_STAGES,
            maxnreg=MAXNREG
        )
        gluon_fn()
        # print('Done')
        # print(f'{c=}')
        # torch.xpu.synchronize()
        # print('After synchro')
    else:
        ### Triton
        triton_fn = lambda: matmul(
            a,
            b,
            c,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            grf_mode=GRF_MODE,
            num_warps=NUM_WARPS,
            num_ctas=NUM_CTAS,
            num_stages=NUM_STAGES,
            maxnreg=MAXNREG
        )
        triton_fn()
        # print('Done')
        # print(f'{c=}')
        # torch.xpu.synchronize()
        # print('After synchro')

    # torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)
    # rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
    # benchmark_suite.assert_close(triton_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='triton to torch')
    # _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)

    # torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)
    # rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
    # benchmark_suite.assert_close(gluon_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='gluon to torch')
    #_, min_ms, max_ms, mean_ms, cv = do_bench(gluon_fn)


