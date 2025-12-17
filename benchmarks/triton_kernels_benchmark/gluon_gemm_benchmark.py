from typing import List
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language.intel import IntelDPASLayout
from utils.dpas_layout_analyzer import calculate_optimal_warps_per_cta, calculate_optimal_rep_clusters


@gluon.constexpr_function
def get_dpas_layout(num_warps: ttgl.constexpr, m_shape: ttgl.constexpr, n_shape: ttgl.constexpr,
                    k_shape: ttgl.constexpr) -> ttgl.constexpr:
    threads_per_warp = 16
    warps_per_cta = calculate_optimal_warps_per_cta(num_warps, m_shape, n_shape)

    return IntelDPASLayout(
        repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, warps_per_cta=warps_per_cta,
        rep_cluster=calculate_optimal_rep_clusters(m_shape, n_shape, k_shape, threads_per_warp,
                                                   warps_per_cta), threads_per_warp=threads_per_warp)


def get_gluon_matmul_autotune_configs() -> List[triton.Config]:
    # configs = [
    #     triton.Config(
    #         {
    #             'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
    #             'NUM_STAGES': s, 'NUM_WARPS': 32
    #         }, num_stages=s, num_warps=32) for s in [2]
    # ]

    configs = [
        triton.Config(
            {
                'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '128',
                'NUM_STAGES': s, 'NUM_WARPS': 64
            }, num_stages=s, num_warps=64) for s in [2]
    ]


    # configs = [
    #     triton.Config(
    #         {
    #             'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
    #             'NUM_STAGES': s, 'NUM_WARPS': 32
    #         }, num_stages=s, num_warps=32) for s in [1, 2, 3]
    # ] + [
    #     triton.Config(
    #         {
    #             'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m,
    #             'NUM_STAGES': s, 'NUM_WARPS': w
    #         }, num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('256', 32), ('128', 64)])
    # ] + [
    #     triton.Config(
    #         {
    #             'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
    #             'NUM_STAGES': s, 'NUM_WARPS': 32
    #         }, num_stages=s, num_warps=32) for s in [2]
    # ] + [
    #     triton.Config(
    #         {
    #             'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m,
    #             'NUM_STAGES': s, 'NUM_WARPS': w
    #         }, num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('256', 32), ('128', 64)])
    # ]
    return configs


@triton.autotune(
    configs=get_gluon_matmul_autotune_configs(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def gluon_matmul_kernel_with_tensor_descriptors(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,
        # Stride variables
        stride_am: ttgl.constexpr, stride_ak: ttgl.constexpr, stride_bk: ttgl.constexpr, stride_bn: ttgl.constexpr,
        stride_cm: ttgl.constexpr, stride_cn: ttgl.constexpr,
        # Meta parameters
        BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,
        GROUP_SIZE_M: ttgl.constexpr,
        # Gluon meta parameters
        NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    layout: ttgl.constexpr = get_dpas_layout(NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=0, k_width=1)
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=1, k_width=2)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ttgl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_desc = ttgl.intel.make_tensor_descriptor(a_ptr, (M, K), (stride_am, stride_ak), (BLOCK_SIZE_M, BLOCK_SIZE_K),
                                               lhs_layout)
    b_desc = ttgl.intel.make_tensor_descriptor(b_ptr, (K, N), (stride_bk, stride_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N),
                                               rhs_layout)
    c_desc = ttgl.intel.make_tensor_descriptor(c_ptr, (M, N), (stride_cm, stride_cn), (BLOCK_SIZE_M, BLOCK_SIZE_N),
                                               layout)

    accumulator = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ttgl.float32, layout=layout)

    # Prefetch first blocks for A and B matrices (pre-loop prefetches)
    for i in range(NUM_STAGES - 1):
        if i * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        a = ttgl.intel.xe.load_2d(a_desc, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = ttgl.intel.xe.load_2d(b_desc, [k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        # Prefetch ahead blocks (pipelining)
        prefetch_k = k + NUM_STAGES - 1
        if prefetch_k * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, prefetch_k * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [prefetch_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        accumulator = ttgl.intel.xe.dpas(a, b, accumulator)

    ttgl.intel.xe.store_2d(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)

def gluon_matmul_kernel_with_tensor_descriptors(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,
        # Stride variables
        stride_am: ttgl.constexpr, stride_ak: ttgl.constexpr, stride_bk: ttgl.constexpr, stride_bn: ttgl.constexpr,
        stride_cm: ttgl.constexpr, stride_cn: ttgl.constexpr,
        # Meta parameters
        BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,
        GROUP_SIZE_M: ttgl.constexpr,
        # Gluon meta parameters
        NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
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
    pass



def get_gluon_matmul_batched_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m,
                'NUM_STAGES': s, 'NUM_WARPS': w
            }, num_stages=s, num_warps=w) for s in [2] for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [2, 3]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 4
            }, num_stages=s, num_warps=4) for s in [2]
    ]
    return configs


@triton.autotune(
    configs=get_gluon_matmul_batched_autotune_configs(),
    key=['B', 'M', 'N', 'K'],
)
@gluon.jit
def gluon_matmul_kernel_with_tensor_descriptors_batched(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,  # pylint: disable=W0613
        # Stride variables
    stride_az: ttgl.constexpr, stride_am: ttgl.constexpr, stride_ak: ttgl.constexpr, stride_bz: ttgl.constexpr,
        stride_bk: ttgl.constexpr, stride_bn: ttgl.constexpr, stride_cz: ttgl.constexpr, stride_cm: ttgl.constexpr,
        stride_cn: ttgl.constexpr,
        # Meta parameters
        BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,
        GROUP_SIZE_M: ttgl.constexpr,
        # Gluon meta parameters
        NUM_STAGES: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    layout: ttgl.constexpr = get_dpas_layout(NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=0, k_width=1)
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=1, k_width=2)

    bid = ttgl.program_id(axis=1)
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ttgl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Calculate batch offsets
    offset_a = bid.to(ttgl.int64) * stride_az
    offset_b = bid.to(ttgl.int64) * stride_bz
    offset_c = bid.to(ttgl.int64) * stride_cz

    a_desc = ttgl.intel.make_tensor_descriptor(a_ptr + offset_a, (M, K), (stride_am, stride_ak),
                                               (BLOCK_SIZE_M, BLOCK_SIZE_K), lhs_layout)
    b_desc = ttgl.intel.make_tensor_descriptor(b_ptr + offset_b, (K, N), (stride_bk, stride_bn),
                                               (BLOCK_SIZE_K, BLOCK_SIZE_N), rhs_layout)
    c_desc = ttgl.intel.make_tensor_descriptor(c_ptr + offset_c, (M, N), (stride_cm, stride_cn),
                                               (BLOCK_SIZE_M, BLOCK_SIZE_N), layout)

    accumulator = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ttgl.float32, layout=layout)

    # Prefetch first blocks for A and B matrices (pre-loop prefetches)
    for i in range(NUM_STAGES - 1):
        if i * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        a = ttgl.intel.xe.load_2d(a_desc, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = ttgl.intel.xe.load_2d(b_desc, [k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        # Prefetch ahead blocks (pipelining)
        prefetch_k = k + NUM_STAGES - 1
        if prefetch_k * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, prefetch_k * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [prefetch_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        accumulator = ttgl.intel.xe.dpas(a, b, accumulator)

    ttgl.intel.xe.store_2d(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)
