
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

@pointwise(size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_79', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 36992.0
    tmp6 = tmp4 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp16, xmask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (55488*y1)), tmp18, xmask)


def get_args():
    arg_0 = rand_strided((128, 192, 17, 17), (55488, 1, 3264, 192), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 192, 1, 1), (192, 1, 192, 192), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 192, 1, 1), (192, 1, 192, 192), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 192, 17, 17), (221952, 289, 17, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((128, 192, 17, 17), (55488, 1, 3264, 192), device='xpu:0', dtype=torch.bool)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_79.run(*args, 24576, 289, grid=grid(24576, 289), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_79.benchmark_all_configs(*args, 24576, 289, grid=grid(24576, 289))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
