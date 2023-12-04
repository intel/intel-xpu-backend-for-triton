
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

@pointwise(size_hints=[536870912], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]})
@triton.jit
def triton_poi_fused_add_slice_backward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 500072448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 30522) % 512
    x2 = (xindex // 15627264)
    x4 = xindex % 15627264
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x4 + (15596742*x2)), tmp3, other=0)
    tmp5 = tl.load(in_ptr2 + (x1 + (511*x2)), tmp3, eviction_policy='evict_last')
    tmp10 = tmp7 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tl.load(in_ptr5 + (x4 + (15596742*x2)), tmp3, other=0)
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.load(in_ptr6 + (x1 + (511*x2)), tmp3, eviction_policy='evict_last', other=0)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 - tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.where(tmp3, tmp19, 0.0)
    tmp21 = tl.where(tmp3, tmp20, tmp11)
    tmp22 = tmp0 + tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)


def get_args():
    arg_0 = rand_strided((32, 512, 30522), (15627264, 30522, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16352, 30522), (30522, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((16352, 1), (1, 1), device='xpu:0', dtype=torch.bool)
    arg_3 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((16352, 30522), (30522, 1), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((16352, 1), (1, 16352), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((32, 512, 30522), (15627264, 30522, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_add_slice_backward_3.run(*args, 500072448, grid=grid(500072448), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_add_slice_backward_3.benchmark_all_configs(*args, 500072448, grid=grid(500072448))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
