from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
from torch._C import _xpu_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
# AttributeError: module 'torch._C' has no attribute '_distributed_c10d'
# empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p

# kernel path: C:\Users\sdp\AppData\Local\Temp\tmp2d_ck1g_\n7\cn7bpauyowgizzwcah7zo2uyqeyppssdju74jq7k32y6w6log3co.py
# Topologically Sorted Source Nodes: [diagonal_attention_scores, setitem, , _generalized_scatter_1, setitem_1, _generalized_scatter_3], Original ATen: [aten.new_zeros, aten.copy]
# Source node to ATen node mapping:
#    => slice_scatter_default, slice_scatter_default_2
#   _generalized_scatter_1 => slice_scatter_default_1
#   _generalized_scatter_3 => select_scatter_default
#   diagonal_attention_scores => full
#   setitem => copy
#   setitem_1 => copy_1
# Graph fragment:
#   %full : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([12, 4, 256, 513], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %slice_4), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %copy, 3, 256, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %slice_scatter_default, 1, 0, -1), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_24, %slice_18), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %copy_1, 2, 256, 9223372036854775807), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default_1, %slice_scatter_default_2, 1, -1), kwargs = {})
triton_poi_fused_copy_new_zeros_3 = async_compile.triton('triton_poi_fused_copy_new_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=20, cc={'architecture': 21479031808, 'driver_version': '1.6.31896', 'gpu_eu_count': 160, 'gpu_subslice_count': 20, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 160, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Arc(TM) B580 Graphics', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 12450455552, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '20.1.0'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C773669FE0C88DDE71A5716565B7045AF2540F52C40A44637E653BAD0B5B739B', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_new_zeros_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 6156) % 4)
    x0 = (xindex % 513)
    x3 = xindex // 24624
    x1 = ((xindex // 513) % 12)
    x5 = (xindex % 6156)
    x6 = xindex // 6156
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = (((656384 + x0 + 513*x3) // 512) % 513)
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + (512*((((656384 + x0 + 513*x3) // 512) % 513)) + 262144*((656384 + x0 + 513*x3) // 262656) + 786432*x1 + 786432*((656384 + x0 + 513*x3) // 787968) + (((x0 + 513*x3) % 512))), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tl.full([1], 3, tl.int64)
    tmp14 = tmp13 < tmp13
    tmp15 = x0
    tmp16 = tl.full([1], 256, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = (((787712 + x0 + 513*x3) // 512) % 513)
    tmp20 = tl.full([1], 512, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21 & tmp18
    tmp23 = tl.load(in_ptr0 + (262144*((((787712 + x0 + 513*x3) // 262656) % 3)) + 786432*((((787712 + x0 + 513*x3 + 787968*x1) // 787968) % 12)) + (((787712 + x0 + 513*x3) % 262656))), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = 0.0
    tmp27 = tl.where(tmp17, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp14, tmp27, tmp28)
    tmp30 = 0.0
    tmp31 = tl.where(tmp14, tmp29, tmp30)
    tmp32 = tl.where(tmp5, tmp12, tmp31)
    tmp33 = tmp0 < tmp13
    tmp34 = x0
    tmp35 = tl.full([1], 256, tl.int64)
    tmp36 = tmp34 >= tmp35
    tmp37 = tmp36 & tmp33
    tmp38 = ((((-256) + x0 + 513*x3 + 262656*x2 + 787968*x1) // 512) % 513)
    tmp39 = tl.full([1], 512, tl.int64)
    tmp40 = tmp38 < tmp39
    tmp41 = tmp40 & tmp37
    tmp42 = tl.load(in_ptr0 + (262144*(((((-256) + x0 + 513*x3 + 262656*x2 + 787968*x1) // 262656) % 36)) + ((((-256) + x0 + 513*x3 + 262656*x2 + 787968*x1) % 262656))), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp37, tmp42, tmp43)
    tmp45 = 0.0
    tmp46 = tl.where(tmp36, tmp44, tmp45)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp33, tmp46, tmp47)
    tmp49 = tl.where(tmp33, tmp48, tmp30)
    tmp50 = tl.where(tmp2, tmp32, tmp49)
    tl.store(out_ptr0 + (x5 + 6176*x6), tmp50, None)
''', device_str='xpu')
