import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch


def get_grid(xnumel, ynumel):
    def grid(META):
        return (
            triton.cdiv(xnumel, META['XBLOCK']),
            triton.cdiv(ynumel, META['YBLOCK']),
            1
        )
    return grid


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=4, num_stages=2),
        #triton.Config({'XBLOCK': 8, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        #triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        #triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        #triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'ynumel'],
    warmup=100,
    rep=200,
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
        in_ptr0, out_ptr0, ynumel, xnumel = args
        grid = get_grid(xnumel, ynumel)
        triton_poi_fused_avg_pool2d_backward_111[grid](
            in_ptr0, out_ptr0, ynumel, xnumel
        )


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    
    # Warmup
    for _ in range(10):
        call(args)
    torch.xpu.synchronize()
    
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100)
    num_gb = 0.1806336
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")