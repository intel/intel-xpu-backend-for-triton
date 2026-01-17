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
    configs = [
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [1, 2, 3]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m,
                'NUM_STAGES': s, 'NUM_WARPS': w
            }, num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256',
                'NUM_STAGES': s, 'NUM_WARPS': 32
            }, num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config(
            {
                'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m,
                'NUM_STAGES': s, 'NUM_WARPS': w
            }, num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('256', 32), ('128', 64)])
    ]
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
    DPAS_LAYOUT: ttgl.constexpr = get_dpas_layout(NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    A_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=DPAS_LAYOUT, operand_index=0, k_width=1)
    B_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=DPAS_LAYOUT, operand_index=1, k_width=2)

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
                                               A_DOT_LAYOUT)
    b_desc = ttgl.intel.make_tensor_descriptor(b_ptr, (K, N), (stride_bk, stride_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N),
                                               B_DOT_LAYOUT)
    c_desc = ttgl.intel.make_tensor_descriptor(c_ptr, (M, N), (stride_cm, stride_cn), (BLOCK_SIZE_M, BLOCK_SIZE_N),
                                               DPAS_LAYOUT)

    accumulator = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ttgl.float32, layout=DPAS_LAYOUT)

    # Prefetch first blocks for A and B matrices (pre-loop prefetches)
    for i in range(NUM_STAGES - 1):
        if i * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        # Prefetch ahead blocks (pipelining)
        prefetch_k = k + NUM_STAGES - 1
        if prefetch_k * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, prefetch_k * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [prefetch_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        a = ttgl.intel.xe.load_2d(a_desc, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = ttgl.intel.xe.load_2d(b_desc, [k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        accumulator = ttgl.intel.xe.dpas(a, b, accumulator)

    ttgl.intel.xe.store_2d(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)


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
    DPAS_LAYOUT: ttgl.constexpr = get_dpas_layout(NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    A_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=DPAS_LAYOUT, operand_index=0, k_width=1)
    B_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=DPAS_LAYOUT, operand_index=1, k_width=2)

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
                                               (BLOCK_SIZE_M, BLOCK_SIZE_K), A_DOT_LAYOUT)
    b_desc = ttgl.intel.make_tensor_descriptor(b_ptr + offset_b, (K, N), (stride_bk, stride_bn),
                                               (BLOCK_SIZE_K, BLOCK_SIZE_N), B_DOT_LAYOUT)
    c_desc = ttgl.intel.make_tensor_descriptor(c_ptr + offset_c, (M, N), (stride_cm, stride_cn),
                                               (BLOCK_SIZE_M, BLOCK_SIZE_N), DPAS_LAYOUT)

    accumulator = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ttgl.float32, layout=DPAS_LAYOUT)

    # Prefetch first blocks for A and B matrices (pre-loop prefetches)
    for i in range(NUM_STAGES - 1):
        if i * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        # Prefetch ahead blocks (pipelining)
        prefetch_k = k + NUM_STAGES - 1
        if prefetch_k * BLOCK_SIZE_K < K:
            ttgl.intel.xe.prefetch_2d(a_desc, [pid_m * BLOCK_SIZE_M, prefetch_k * BLOCK_SIZE_K])
            ttgl.intel.xe.prefetch_2d(b_desc, [prefetch_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        a = ttgl.intel.xe.load_2d(a_desc, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = ttgl.intel.xe.load_2d(b_desc, [k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])

        accumulator = ttgl.intel.xe.dpas(a, b, accumulator)

    ttgl.intel.xe.store_2d(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)
