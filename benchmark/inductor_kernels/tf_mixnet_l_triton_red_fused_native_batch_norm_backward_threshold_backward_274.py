
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@reduction(
    size_hints=[262144, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_274', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_274(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 150528
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (393216*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((12544*x1) + (2408448*((r2 + (2048*x0)) // 12544)) + ((r2 + (2048*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((128, 192, 112, 112), (2408448, 1, 21504, 192), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((128, 192, 112, 112), (2408448, 12544, 112, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((192, 784), (784, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_274.run(*args, 150528, 2048, grid=grid(150528), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_native_batch_norm_backward_threshold_backward_274.benchmark_all_configs(*args, 150528, 2048, grid=grid(150528))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
