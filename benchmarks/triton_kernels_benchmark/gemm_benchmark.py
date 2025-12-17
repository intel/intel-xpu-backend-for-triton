"""
Gemm benchmark
============================

This benchmark is come from the Triton tutorial 10-experimental-block-pointer.py
To compare the performance to XeTLA kernel.

"""
from typing import Callable, List, Optional
import os

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark import xetla_kernel
from triton_kernels_benchmark import cutlass_kernel
from gluon_gemm_benchmark import gluon_matmul_kernel_with_tensor_descriptors, gluon_matmul_kernel_with_tensor_descriptors_batched


def get_matmul_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '128'},
            num_stages=s, num_warps=64) for s in [2]
    ]

    # configs = [
    #     triton.Config(
    #         {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
    #         num_stages=s, num_warps=32) for s in [1, 2, 3]
    # ] + [
    #     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
    #                   num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('256', 32), ('128', 64)])
    # ] + [
    #     triton.Config(
    #         {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
    #         num_stages=s, num_warps=32) for s in [2]
    # ] + [
    #     triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
    #                   num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('256', 32), ('128', 64)])
    # ]
    return configs


@triton.autotune(
    configs=get_matmul_autotune_configs(),
    key=['M', 'N', 'K'],
)
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

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
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


def get_matmul_batched_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2] for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '256'},
            num_stages=s, num_warps=4) for s in [2]
    ]
    return configs


# pylint: disable=unused-argument
@triton.autotune(
    configs=get_matmul_batched_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_block_pointers_batched(
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

    a_block_ptr = tl.make_block_ptr(base=a_ptr + offset_a, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(tl.float32)

    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) launches the above kernel.
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    matmul_kernel: Callable,
    matmul_kernel_batched: Callable,
    transpose_a=False,
    transpose_b=False,
    is_gluon=False
):
    a_major, a_minor = -2, -1
    if transpose_a:
        a_major, a_minor = a_minor, a_major
    b_minor, b_major = -2, -1
    if transpose_b:
        b_major, b_minor = b_minor, b_major

    assert a.shape[a_minor] == b.shape[b_minor], 'Incompatible dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    assert b.is_contiguous(), 'Matrix B must be contiguous'
    M, N, K = a.shape[a_major], b.shape[b_major], a.shape[a_minor]
    # Check constraints.
    if len(a.shape) == 3 and len(b.shape) == 3:
        assert a.shape[0] == b.shape[0], 'Incompatible Batch dimension'
        B = a.shape[0]
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            B,
        )
        matmul_kernel_batched[grid](a, b, c, B, M, N, K, a.stride(0), a.stride(a_major), a.stride(a_minor), b.stride(0),
                                    b.stride(b_minor), b.stride(b_major), c.stride(0), c.stride(1), c.stride(2))
    elif len(a.shape) == 2 and len(b.shape) == 2:
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        # M=4
        # N=12288
        # BLOCK_SIZE_M=8
        # BLOCK_SIZE_N=512

        BLOCK_SIZE_M = 8
        BLOCK_SIZE_N = 512
        #BLOCK_SIZE_K = 64
        #GROUP_SIZE_M = 1
        #grf_mode = '128'
        #num_warps = 64
        #num_ctas = 1
        #num_stages = 2
        #maxnreg = None

        # NUM_STAGES = num_stages
        # NUM_WARPS = num_warps

        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)

        if is_gluon:
            matmul_kernel[grid](a, b, c) #, M, N, K, a.stride(a_major), a.stride(a_minor), b.stride(b_minor),
                                #b.stride(b_major), c.stride(0), c.stride(1)) #, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_STAGES, NUM_WARPS, grf_mode, num_stages, num_warps)
        else:
            matmul_kernel[grid](a, b, c, M, N, K, a.stride(a_major), a.stride(a_minor), b.stride(b_minor),
                                b.stride(b_major), c.stride(0), c.stride(1)) #, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, grf_mode, num_stages, num_warps)

    else:
        assert False, 'Input matrixs dimensions mismatch'
    return c


def get_shapes(B, M, N, K, transpose_a, transpose_b):
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


X_VALS = [  #
    # [1, 1, 1024, 4096],
    # [1, 1, 4096, 4096],
    # [1, 1, 4096, 14336],
    # [1, 1, 6144, 4096],
    # [1, 1, 13824, 5120],
    # [1, 1, 14336, 4096],
    # [1, 1, 28672, 4096],
    # [1, 1, 128256, 4096],
    [1, 4, 12288, 4096],
    # [1, 8, 1024, 4096],
    # [1, 8, 4096, 4096],
    # [1, 8, 4096, 14336],
    # [1, 8, 6144, 4096],
    # [1, 8, 14336, 4096],
    # [1, 8, 28672, 4096],
    # [1, 8, 128256, 4096],
    # [1, 512, 8192, 8192],
    # [1, 512, 8192, 32768],
    # [1, 512, 32768, 8192],
    # [1, 1024, 1024, 1024],
    # [1, 1024, 8192, 16384],
    # [1, 1024, 8192, 28672],
    # [1, 2048, 2048, 2048],
    # [1, 3072, 3072, 4096],  # FIXME: Remove this case when gemm_streamk_benchmark can get better performance
    # [1, 4096, 4096, 4096],
    # [1, 4096, 8192, 16384],
    # [1, 8192, 1024, 16384],
    # [1, 8192, 4096, 4096],
    # [1, 8192, 4096, 16384],
    # [1, 8192, 8192, 8192],
    # [1, 16384, 1024, 8192],
    # [1, 16384, 4096, 8192],
    # [1, 16384, 8192, 1024],
    # [1, 16384, 8192, 4096],
    # [4, 32768, 128, 4096],
    # [4, 32768, 4096, 128],
    # [32, 4096, 128, 4096],
    # [4096, 8, 128, 16384],
    # [4096, 8, 16384, 128],
]

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    # x_val: (B, M, N, K)
    B, M, N, K = x_val
    # a: (B, M, K) bfloat16
    # b: (B, N, K) bfloat16
    # c: (B, M, N) float32
    # pytorch reference: (B, M, N) float32
    required_memory = B * M * K * 2 + B * N * K * 2 + 2 * B * M * N * 4
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    transpose_a=False,
    transpose_b=False,
    triton_matmul_kernel=matmul_kernel_with_block_pointers,
    triton_matmul_kernel_batched=matmul_kernel_with_block_pointers_batched,
    gluon_matmul_kernel=gluon_matmul_kernel_with_tensor_descriptors,
    gluon_matmul_kernel_batched=gluon_matmul_kernel_with_tensor_descriptors_batched,
    plot_name='matmul-performance',
):
    """
    Returns a Mark object containing a Benchmark object constructed at runtime and parameterized by the provided option values.
    The benchmark can then be executed by calling the :code:`.run` method on the return value.
    """
    supported_providers = {
        'gluon': 'Gluon',
        'triton': 'Triton',
        #'onednn': 'OneDNN',
    }
    # use_cutlass
    # if not (transpose_a or transpose_b):
    #     if torch.xpu.get_device_name() != 'Intel(R) Arc(TM) Graphics':
    #         # FIXME: enable cutlass on LNL
    #         supported_providers['cutlass'] = 'CUTLASS'
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    # Benchmark Performance
    # pylint: disable=too-many-branches
    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['B', 'M', 'N', 'K'],
            # different possible values for `x_name`
            x_vals=X_VALS,
            line_arg='provider',
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=list(providers.keys()),
            # label name for the lines
            line_names=list(providers.values()),
            # line styles
            styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
            ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
            plot_name=plot_name,
            # name for the plot. Used also as a file name for saving the plot.
            args={},
        ))
    def benchmark(B, M, N, K, provider):
        # Maximum across onednn=600, triton=800, xetla=10, cutlass=600
        do_bench = benchmark_suite.get_do_bench(n_warmup=800, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        a_shape, b_shape = get_shapes(B, M, N, K, transpose_a=transpose_a, transpose_b=transpose_b)

        torch.manual_seed(0)
        a = torch.rand(a_shape, device='xpu', dtype=torch.bfloat16)
        b = torch.rand(b_shape, device='xpu', dtype=torch.bfloat16)

        torch_a = a
        if transpose_a:
            torch_a = torch.transpose(torch_a, -2, -1)

        torch_b = b
        if transpose_b:
            torch_b = torch.transpose(torch_b, -2, -1)

        if provider == 'onednn':
            _, min_ms, max_ms, mean_ms, cv = do_bench(lambda: torch.matmul(torch_a, torch_b))

        elif provider in ('triton', 'gluon'):
            if len(a.shape) != len(b.shape):
                raise AssertionError(f'Incompatible sizes {len(a.shape)} and {len(b.shape)}', )
            if len(a.shape) == 3:
                c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
            elif len(a.shape) == 2:
                c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
            else:
                raise AssertionError(f'Unexpected shape of length {len(a.shape)}')

            ir = """
#loc1 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":72:0)
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 64], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#loc37 = loc("a_ptr"(#loc1))
#loc38 = loc("b_ptr"(#loc1))
#loc39 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 64 : i32, ttg.target = "xpu:pvc", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.supported_sg_sizes = dense<[16, 32]> : tensor<2xi32>, ttig.target_arch = "spir64"} {
  tt.func public @gluon_matmul_kernel_with_tensor_descriptors(%a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %true = arith.constant true loc(#loc2)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc)
    %c512_i32 = arith.constant 512 : i32 loc(#loc)
    %c8_i32 = arith.constant 8 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst = arith.constant dense<0.000000e+00> : tensor<8x512xf32, #mma> loc(#loc)
    %c12288_i64 = arith.constant 12288 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c4_i64 = arith.constant 4 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c4096_i64 = arith.constant 4096 : i64 loc(#loc)
    %c24_i32 = arith.constant 24 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %pid = tt.get_program_id x : i32 loc(#loc40)
    %group_id = arith.divsi %pid, %c24_i32 : i32 loc(#loc41)
    %group_size_m = arith.subi %c1_i32, %group_id : i32 loc(#loc42)
    %group_size_m_0 = arith.minsi %group_size_m, %c1_i32 : i32 loc(#loc43)
    %pid_m = arith.remsi %pid, %c24_i32 : i32 loc(#loc44)
    %pid_m_1 = arith.remsi %pid_m, %group_size_m_0 : i32 loc(#loc45)
    %pid_m_2 = arith.addi %group_id, %pid_m_1 : i32 loc(#loc46)
    %pid_n = arith.divsi %pid_m, %group_size_m_0 : i32 loc(#loc47)
    %a_desc = tt.make_tensor_ptr %a_ptr, [%c4_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>> loc(#loc48)
    %b_desc = tt.make_tensor_ptr %b_ptr, [%c4096_i64, %c12288_i64], [%c12288_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>> loc(#loc49)
    %c_desc = tt.make_tensor_ptr %c_ptr, [%c4_i64, %c12288_i64], [%c12288_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x512xf32, #mma>> loc(#loc50)
    scf.if %true {
      %3 = arith.muli %pid_m_2, %c8_i32 : i32 loc(#loc15)
      %4 = tt.make_tensor_ptr %a_ptr, [%c4_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xbf16>> loc(#loc16)
      ttig.prefetch %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<8x64xbf16>> loc(#loc16)
      %5 = arith.muli %pid_n, %c512_i32 : i32 loc(#loc17)
      %6 = tt.make_tensor_ptr %b_ptr, [%c4096_i64, %c12288_i64], [%c12288_i64, %c1_i64], [%c0_i32, %5] {order = array<i32: 1, 0>} : <tensor<64x512xbf16>> loc(#loc18)
      ttig.prefetch %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x512xbf16>> loc(#loc18)
    } else {
    } loc(#loc14)
    %accumulator = scf.for %k = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%accumulator_3 = %cst) -> (tensor<8x512xf32, #mma>)  : i32 {
      %a = arith.muli %pid_m_2, %c8_i32 : i32 loc(#loc52)
      %a_4 = arith.muli %k, %c64_i32 : i32 loc(#loc53)
      %a_5 = tt.advance %a_desc, [%a, %a_4] : <tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>> loc(#loc54)
      %a_6 = tt.load %a_5 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>> loc(#loc54)
      %b = arith.muli %pid_n, %c512_i32 : i32 loc(#loc55)
      %b_7 = tt.advance %b_desc, [%a_4, %b] : <tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>> loc(#loc56)
      %b_8 = tt.load %b_7 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>> loc(#loc56)
      %prefetch_k = arith.addi %k, %c1_i32 : i32 loc(#loc57)
      %3 = arith.muli %prefetch_k, %c64_i32 : i32 loc(#loc26)
      %4 = arith.cmpi slt, %3, %c4096_i32 : i32 loc(#loc27)
      scf.if %4 {
        %5 = tt.make_tensor_ptr %a_ptr, [%c4_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%a, %3] {order = array<i32: 1, 0>} : <tensor<8x64xbf16>> loc(#loc29)
        ttig.prefetch %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<8x64xbf16>> loc(#loc29)
        %6 = tt.make_tensor_ptr %b_ptr, [%c4096_i64, %c12288_i64], [%c12288_i64, %c1_i64], [%3, %b] {order = array<i32: 1, 0>} : <tensor<64x512xbf16>> loc(#loc30)
        ttig.prefetch %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x512xbf16>> loc(#loc30)
      } else {
      } loc(#loc28)
      %accumulator_9 = tt.dot %a_6, %b_8, %accumulator_3, inputPrecision = tf32 : tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x512xf32, #mma> loc(#loc58)
      scf.yield %accumulator_9 : tensor<8x512xf32, #mma> loc(#loc32)
    } loc(#loc51)
    %0 = arith.muli %pid_m_2, %c8_i32 : i32 loc(#loc33)
    %1 = arith.muli %pid_n, %c512_i32 : i32 loc(#loc34)
    %2 = tt.advance %c_desc, [%0, %1] : <tensor<8x512xf32, #mma>> loc(#loc35)
    tt.store %2, %accumulator {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<8x512xf32, #mma>> loc(#loc35)
    tt.return loc(#loc36)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":111:30)
#loc3 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":90:26)
#loc4 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":94:22)
#loc5 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":96:44)
#loc6 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":96:57)
#loc7 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":97:34)
#loc8 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":97:54)
#loc9 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":97:27)
#loc10 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":98:40)
#loc11 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":101:47)
#loc12 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":103:47)
#loc13 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":105:47)
#loc14 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":111:11)
#loc15 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":112:55)
#loc16 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":112:46)
#loc17 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":113:73)
#loc18 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":113:46)
#loc19 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":115:22)
#loc20 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":116:51)
#loc21 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":116:69)
#loc22 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":116:42)
#loc23 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":117:69)
#loc24 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":117:42)
#loc25 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":120:38)
#loc26 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":121:24)
#loc27 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":121:39)
#loc28 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":121:11)
#loc29 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":122:46)
#loc30 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":123:46)
#loc31 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":125:47)
#loc32 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":125:8)
#loc33 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":127:44)
#loc34 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":127:66)
#loc35 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":127:81)
#loc36 = loc("/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gluon_gemm_benchmark.py":127:4)
#loc40 = loc("pid"(#loc3))
#loc41 = loc("group_id"(#loc4))
#loc42 = loc("group_size_m"(#loc5))
#loc43 = loc("group_size_m"(#loc6))
#loc44 = loc("pid_m"(#loc7))
#loc45 = loc("pid_m"(#loc8))
#loc46 = loc("pid_m"(#loc9))
#loc47 = loc("pid_n"(#loc10))
#loc48 = loc("a_desc"(#loc11))
#loc49 = loc("b_desc"(#loc12))
#loc50 = loc("c_desc"(#loc13))
#loc51 = loc("accumulator"(#loc19))
#loc52 = loc("a"(#loc20))
#loc53 = loc("a"(#loc21))
#loc54 = loc("a"(#loc22))
#loc55 = loc("b"(#loc23))
#loc56 = loc("b"(#loc24))
#loc57 = loc("prefetch_k"(#loc25))
#loc58 = loc("accumulator"(#loc31))

    """
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
                f.write(ir)
                f.flush()
                test_kernel = triton.compile(f.name)

            #kernel = triton_matmul_kernel if provider == 'triton' else gluon_matmul_kernel
            kernel = triton_matmul_kernel if provider == 'triton' else test_kernel


            batched_kernel = triton_matmul_kernel_batched if provider == 'triton' else gluon_matmul_kernel_batched

            matmul_fn = lambda: matmul(
                a,
                b,
                c,
                matmul_kernel=kernel,
                matmul_kernel_batched=batched_kernel,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                is_gluon=provider=='gluon'
            )
            torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)
            rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
            benchmark_suite.assert_close(matmul_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg=f'{provider} to torch')
            _, min_ms, max_ms, mean_ms, cv = do_bench(matmul_fn)

        elif provider == 'xetla':
            if B == 1:
                c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                cnt = torch.zeros((M, N), device='xpu', dtype=torch.int32)
            else:
                c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                cnt = torch.zeros((B, M, N), device='xpu', dtype=torch.int32)
            name = f'gemm_shape_{B}_{M}_{K}_{N}'
            # FIXME: Use gemm_streamk_benchmark.py when Triton streamk can get
            # better performance.
            if (B, M, N, K) == (1, 3072, 3072, 4096):
                name = 'gemm_streamk_shape_3072_4096_3072'
            func = getattr(xetla_kernel, name)

            def xetla_func_with_acc_allocation():
                # allocating `acc` matrix on every function call, to be as similar as
                # possible to the triton kernel, which also does this on every call.
                if B == 1:
                    acc = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                else:
                    acc = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                return func(a, b, c, acc, cnt)

            xetla_fn = xetla_func_with_acc_allocation
            torch_fn = lambda: torch.matmul(a, b).to(torch.float32)

            # benchmark_suite.assert_close(xetla_fn, torch_fn, atol=1e-4, rtol=1.0, err_msg='xetla to torch')
            _, min_ms, max_ms, mean_ms, cv = do_bench(xetla_fn)

        elif provider == 'cutlass':
            name = 'gemm'
            func = getattr(cutlass_kernel, name)

            # Special case where the b matrix needs to be transposed (see: `./cutlass_kernel/gemm/input_gemm.in`)
            if (B, M, N, K) == (1, 1, 1024, 4096):
                _, b_shape = get_shapes(B, M, N, K, transpose_a=False, transpose_b=True)
                b = torch.reshape(b, b_shape)
                torch_b = b
                torch_b = torch.transpose(torch_b, -2, -1)

            def cutlass_invoker():
                if B == 1:
                    c = torch.zeros((M, N), device='xpu', dtype=torch.float32)
                else:
                    c = torch.zeros((B, M, N), device='xpu', dtype=torch.float32)
                func(a, b, c, M, N, K, B)
                return c

            cutlass_fn = cutlass_invoker
            torch_fn = lambda: torch.matmul(torch_a, torch_b).to(torch.float32)

            rtol = 1e-2 if a.dtype == torch.bfloat16 else 1e-3
            benchmark_suite.assert_close(cutlass_fn, torch_fn, atol=1e-4, rtol=rtol, err_msg='cutlass to torch')
            _, min_ms, max_ms, mean_ms, cv = do_bench(cutlass_fn)

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        tflops = lambda ms: 2 * B * M * N * K * (1e-12) / (ms * 1e-3)
        gbps = lambda ms: B * (2 * (M * K + K * N) + 4.0 * (M * N)) * (1e-9) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    _benchmark = get_benchmark(
        transpose_a=(os.getenv('TRANSPOSE_A', '0') == '1'),
        transpose_b=(os.getenv('TRANSPOSE_B', '0') == '1'),
    )
    _benchmark.run(show_plots=False, print_data=True)
