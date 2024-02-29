
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

@pointwise(size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_col2im_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 55296
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 9
    y2 = (yindex // 3456)
    y5 = yindex % 3456
    y4 = (yindex // 9)
    tmp0 = tl.load(in_ptr0 + (x3 + (512*y0)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, YBLOCK])
    tmp5 = tl.load(in_ptr2 + (y5 + (3456*x3) + (1769472*y2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 520, tmp0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp1 + (520*y4), [XBLOCK, YBLOCK])), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((9, 512, 1, 1), (512, 1, 512, 512), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((1, 1), (1, 1), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((49152, 64, 9), (576, 9, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((16, 384, 520, 1), (199680, 520, 1, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_col2im_25.run(*args, 55296, 512, grid=grid(55296, 512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_col2im_25.benchmark_all_configs(*args, 55296, 512, grid=grid(55296, 512))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
