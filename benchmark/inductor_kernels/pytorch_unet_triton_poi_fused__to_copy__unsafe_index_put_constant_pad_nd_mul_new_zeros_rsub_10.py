
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_put_constant_pad_nd_mul_new_zeros_rsub_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_put_constant_pad_nd_mul_new_zeros_rsub_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39239680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 958) % 640
    x0 = xindex % 958
    x3 = (xindex // 958)
    x2 = (xindex // 613120)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 320, tmp0)
    tmp3 = tl.where(tmp2 < 0, tmp2 + 479, tmp2)
    tmp4 = x0
    tmp5 = tl.full([1], 959, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr2 + (39280640 + x0 + (959*x3)), tmp6, other=0).to(tl.float32)
    tmp8 = tl.where(tmp6, tmp7, 0.0)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tl.where(tmp14 < 0, tmp14 + 320, tmp14)
    tmp16 = 1.0
    tmp17 = tmp16 - tmp12
    tmp18 = tmp11 * tmp17
    tmp20 = tl.where(tmp19 < 0, tmp19 + 479, tmp19)
    tmp21 = tmp16 - tmp10
    tmp22 = tmp9 * tmp21
    tmp23 = tmp22 * tmp12
    tmp24 = tmp22 * tmp17
    tl.atomic_add(out_ptr0 + (tmp3 + (479*tmp1) + (153280*x2)), tmp13, None)
    tl.atomic_add(out_ptr1 + (tmp3 + (479*tmp15) + (153280*x2)), tmp18, None)
    tl.atomic_add(out_ptr2 + (tmp20 + (479*tmp1) + (153280*x2)), tmp23, None)
    tl.atomic_add(out_ptr3 + (tmp20 + (479*tmp15) + (153280*x2)), tmp24, None)


def get_args():
    arg_0 = rand_strided((640, 1), (1, 1), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((958,), (1,), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((1, 128, 640, 959), (78561280, 613760, 959, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((958,), (1,), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((640, 1), (1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((640, 1), (1, 1), device='xpu:0', dtype=torch.int64)
    arg_6 = rand_strided((958,), (1,), device='xpu:0', dtype=torch.int64)
    arg_7 = rand_strided((1, 64, 320, 479), (9809920, 153280, 479, 1), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((1, 64, 320, 479), (9809920, 153280, 479, 1), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((1, 64, 320, 479), (9809920, 153280, 479, 1), device='xpu:0', dtype=torch.float32)
    arg_10 = rand_strided((1, 64, 320, 479), (9809920, 153280, 479, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__to_copy__unsafe_index_put_constant_pad_nd_mul_new_zeros_rsub_10.run(*args, 39239680, grid=grid(39239680), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__to_copy__unsafe_index_put_constant_pad_nd_mul_new_zeros_rsub_10.benchmark_all_configs(*args, 39239680, grid=grid(39239680))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
