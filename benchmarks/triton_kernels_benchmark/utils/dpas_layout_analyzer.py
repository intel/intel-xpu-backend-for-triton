from functools import wraps
from triton._C.libtriton import intel

from triton.experimental.gluon.language.intel.xpu.xe import get_dpas_capabilities
from triton.language.core import TRITON_BUILTIN


def allow_in_kernel(fn):
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


@allow_in_kernel
def calculate_optimal_warps_per_cta(num_warps, m_shape, n_shape):
    ret_shape = [m_shape, n_shape]
    dpas_cap = get_dpas_capabilities()
    return intel.calculate_warps_per_tile(capRepeatCount=dpas_cap['repeatCount'],
                                          capExecutionSize=dpas_cap['executionSize'], shape=ret_shape,
                                          numWarps=num_warps)


@allow_in_kernel
def calculate_optimal_rep_clusters(block_m, block_n, block_k, threads_per_warp, warps_per_cta):
    dtype_bitwidth = 16  # bf16  TODO: auto detect
    is_fp8 = dtype_bitwidth == 8
    dpas_cap = get_dpas_capabilities()
    cap_repeat_count = dpas_cap['repeatCount']
    cap_systolic_depth = dpas_cap['systolicDepth']
    cap_execution_size = dpas_cap['executionSize']
    ops_per_chan = int(dpas_cap['opsChanBitWidths'] / dtype_bitwidth)

    ret_shape = [block_m, block_n]
    a_shape = [block_m, block_k]
    b_shape = [block_k, block_n]

    rep_cluster = intel.calculate_rep_cluster(cap_repeat_count=cap_repeat_count, cap_systolic_depth=cap_systolic_depth,
                                              cap_execution_size=cap_execution_size, ops_per_chan=ops_per_chan,
                                              ret_shape=ret_shape, threads_per_warp=threads_per_warp,
                                              a_bitwidth=dtype_bitwidth, is_fp8=is_fp8, a_shape=a_shape,
                                              b_shape=b_shape, warps_per_tile=warps_per_cta)

    return rep_cluster
