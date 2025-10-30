import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch.profiler import profile, ProfilerActivity
import torch._inductor.config as config


import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ['TRITON_INTERPRET'] = '0'
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['LLVM_IR_ENABLE_DUMP'] = '1'

@triton_heuristics.pointwise(
    size_hints={'x': 268435456},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=20, cc={'architecture': 21479031808, 'device_id': 57867, 'driver_version': '1.6.33578+13', 'gpu_eu_count': 160, 'gpu_subslice_count': 20, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 160, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Arc(TM) B580 Graphics', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 12168933376, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '20.1.0'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'autotune_pointwise': True, 'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4F98A0CCE2C4E09763E3797B17267BD555C1ED52C2FEA9E881716C4F50E8E044', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'kernel_num_gb': 0.48500736, 'kernel_flop': 0},
    min_elem_per_thread=4,
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 177020928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 147)
    x2 = ((xindex // 9408) % 147)
    x3 = xindex // 1382976
    x4 = ((xindex // 64) % 21609)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp35 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp48 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp1 = (tmp0).to(tl.int32)
    tmp2 = (0)
    tmp3 = tmp1 >= tmp2
    tmp4 = tmp1 < tmp2
    tmp5 = tl.where(tmp4, tmp1 + 9, tmp1)
    tmp7 = 0.0
    tmp8 = tmp5 % 9
    tmp9 = tmp5 // 9
    tmp10 = (tmp9 < x4).to(tl.int1)
    tmp11 = tmp3 & tmp10
    tmp13 = (tmp12).to(tl.int32)
    tmp14 = tmp13 >= tmp2
    tmp15 = tmp13 < tmp2
    tmp16 = tl.where(tmp15, tmp13 + 9, tmp13)
    tmp18 = tmp16 % 9
    tmp19 = tmp16 // 9
    tmp20 = (tmp19 < x4).to(tl.int1)
    tmp21 = tmp14 & tmp20
    tmp22 = tmp11 & tmp21
    tmp23 = tmp8 == (0)
    tmp24 = tmp18 == (0)
    tmp25 = tmp23 & tmp24
    tmp26 = tmp22 & tmp25
    tmp27 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp26).to(tl.float32)
    tmp28 = tmp6 - tmp27
    tmp29 = tl.where(tmp26, tmp28, tmp7)
    tmp31 = (tmp30).to(tl.int32)
    tmp32 = tmp31 >= tmp2
    tmp33 = tmp31 < tmp2
    tmp34 = tl.where(tmp33, tmp31 + 9, tmp31)
    tmp36 = tmp34 % 9
    tmp37 = tmp34 // 9
    tmp38 = (tmp37 < x4).to(tl.int1)
    tmp39 = tmp32 & tmp38
    tmp40 = tmp39 & tmp21
    tmp41 = tmp36 == (0)
    tmp42 = tmp41 & tmp24
    tmp43 = tmp40 & tmp42
    tmp44 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp43).to(tl.float32)
    tmp45 = tmp35 - tmp44
    tmp46 = tl.where(tmp43, tmp45, tmp7)
    tmp47 = tmp29 + tmp46
    tmp49 = (tmp48).to(tl.int32)
    tmp50 = tmp49 >= tmp2
    tmp51 = tmp49 < tmp2
    tmp52 = tl.where(tmp51, tmp49 + 9, tmp49)
    tmp53 = tmp52 % 9
    tmp54 = tmp52 // 9
    tmp55 = (tmp54 < x4).to(tl.int1)
    tmp56 = tmp50 & tmp55
    tmp57 = tmp11 & tmp56
    tmp58 = tmp23 & tmp41
    tmp59 = tmp57 & tmp58
    tmp60 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp59).to(tl.float32)
    tmp61 = tmp6 - tmp60
    tmp62 = tl.where(tmp59, tmp61, tmp7)
    tmp63 = tmp47 + tmp62
    tmp64 = tmp39 & tmp56
    tmp65 = tmp41 & tmp41
    tmp66 = tmp64 & tmp65
    tmp67 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp66).to(tl.float32)
    tmp68 = tmp35 - tmp67
    tmp69 = tl.where(tmp66, tmp68, tmp7)
    tmp70 = tmp63 + tmp69
    tmp71 = tmp22 & tmp56
    tmp72 = tmp25 & tmp41
    tmp73 = tmp71 & tmp72
    tmp74 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp73).to(tl.float32)
    tmp75 = tmp17 - tmp74
    tmp76 = tl.where(tmp73, tmp75, tmp7)
    tmp77 = tmp70 + tmp76
    tmp78 = tmp40 & tmp56
    tmp79 = tmp42 & tmp41
    tmp80 = tmp78 & tmp79
    tmp81 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), tmp80).to(tl.float32)
    tmp82 = tmp17 - tmp81
    tmp83 = tl.where(tmp80, tmp82, tmp7)
    tmp84 = tmp77 + tmp83
    tl.store(out_ptr0 + (x7), tmp84, None)


def get_args():
    arg_0 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.int8)
    arg_1 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.float16)
    arg_2 = rand_strided((128, 64, 147, 147), (1382976, 1, 9408, 64), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 177020928,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    
    print("Warming up kernel...")
    for _ in range(10):
        call(args)
    torch.xpu.synchronize()

    print("\n=== Benchmark ===")
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.48500736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
    
    print("\n=== PyTorch Profiler ===")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            call(args)
            torch.xpu.synchronize()
    
    print("\n--- Profiler Summary (sorted by XPU time) ---")
    print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=20))
    
    prof.export_chrome_trace("triton_kernel_xpu_profile.json")
    print("\nProfile trace exported to: triton_kernel_xpu_profile.json")
