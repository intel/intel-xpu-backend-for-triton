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
    size_hints={'y': 4, 'x': 256}, tile_hint=TileHint.DEFAULT,
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
    y2 = yindex // 1225
    y5 = (yindex % 1225)
    idx = x3 + 288*y5 + 352800*y2
    mask = xmask & ymask
    tmp0 = tl.load(in_ptr0 + idx, mask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + idx + 288, mask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + idx + 576, mask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + idx + 10080, mask).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + idx + 10368, mask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + idx + 10656, mask).to(tl.float32)
    tmp6 = tl.load(in_ptr0 + idx + 20160, mask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + idx + 20448, mask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + idx + 20736, mask).to(tl.float32)
    result = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8
    tl.store(out_ptr0 + (y5 + 1280*x3 + 368640*y2), result, mask)


def get_args():
    arg_0 = rand_strided((128, 288, 35, 35), (352800, 1, 10080, 288), device='xpu:0', dtype=torch.float16)
    arg_1 = rand_strided((128, 288, 35, 35), (368640, 1280, 35, 1), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, 156800, 288


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