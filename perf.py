import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=20, cc={'architecture': 21479031808, 'device_id': 57867, 'driver_version': '1.6.33578+13', 'gpu_eu_count': 160, 'gpu_subslice_count': 20, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 160, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Arc(TM) B580 Graphics', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 12168933376, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '20.1.0'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_111', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '4F98A0CCE2C4E09763E3797B17267BD555C1ED52C2FEA9E881716C4F50E8E044', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 180633600, 'x': 812851200}, 'kernel_num_gb': 0.1806336, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_backward_111(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 156800
    xnumel = 288
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 35)
    y1 = ((yindex // 35) % 35)
    y2 = yindex // 1225
    y5 = (yindex % 1225)
    tmp0 = tl.load(in_ptr0 + (x3 + 288*((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (x3 + 288*((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (x3 + 288*((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (x3 + 288*((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (x3 + 288*((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp37 = tl.load(in_ptr0 + (x3 + 288*((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tl.load(in_ptr0 + (x3 + 288*((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr0 + (x3 + 288*((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp54 = tl.load(in_ptr0 + (x3 + 288*((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))))) + 10080*((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) <= ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35))))) + ((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) * (((-1) + ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))) < (2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))))) + 352800*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = (tmp0 / 9)
    tmp2 = ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))
    tmp3 = ((35) * ((35) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (35)))
    tmp4 = tmp2 < tmp3
    tmp5 = ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))
    tmp6 = ((35) * ((35) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (35)))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = (tmp11 / 9)
    tmp13 = 1 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = (tmp18 / 9)
    tmp20 = 2 + ((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = (tmp25 / 9)
    tmp27 = 1 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = (tmp32 / 9)
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = (tmp37 / 9)
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = (tmp42 / 9)
    tmp44 = 2 + ((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = (tmp49 / 9)
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = (tmp54 / 9)
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (y5 + 1280*x3 + 368640*y2), tmp58, xmask & ymask)


def get_args():
    arg_0 = rand_strided((128, 288, 35, 35), (352800, 1, 10080, 288), device='xpu:0', dtype=torch.float16)
    arg_1 = rand_strided((128, 288, 35, 35), (368640, 1280, 35, 1), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, 156800, 288,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_backward_111.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_avg_pool2d_backward_111.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.1806336
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")

