
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

@pointwise(size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_relu_threshold_backward_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]})
@triton.jit
def triton_poi_fused__to_copy_convolution_relu_threshold_backward_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (193600*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = triton_helpers.maximum(0, tmp3)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + (3025*y0) + (387200*y1)), tmp4, xmask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (193600*y1)), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((32, 64, 55, 55), (193600, 1, 3520, 64), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64,), (1,), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 64, 55, 55), (387200, 3025, 55, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((32, 64, 55, 55), (193600, 1, 3520, 64), device='xpu:0', dtype=torch.bool)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__to_copy_convolution_relu_threshold_backward_8.run(*args, 2048, 3025, grid=grid(2048, 3025), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__to_copy_convolution_relu_threshold_backward_8.benchmark_all_configs(*args, 2048, 3025, grid=grid(2048, 3025))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
