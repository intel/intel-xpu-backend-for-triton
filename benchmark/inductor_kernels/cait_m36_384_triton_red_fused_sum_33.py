
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
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_sum_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 23699
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 14
    x1 = (xindex // 14)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (23699*x0)
        tmp1 = tl.full([1, 1], 331776, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((331776*x1) + ((r2 + (23699*x0)) % 331776)), rmask & tmp2 & xmask, other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + ((331776*x1) + ((r2 + (23699*x0)) % 331776)), rmask & tmp2 & xmask, other=0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tl.load(in_ptr2 + ((576*x1) + (((r2 + (23699*x0)) // 576) % 576)), rmask & tmp2 & xmask, other=0)
        tmp9 = tmp6 * tmp8
        tmp10 = tmp7 - tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.where(tmp2, tmp11, 0)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)


def get_args():
    arg_0 = rand_strided((16, 331776), (331776, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 16, 576, 576), (5308416, 331776, 576, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((1, 16, 576, 1), (9216, 576, 1, 9216), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 1, 1, 16, 14), (224, 224, 224, 14, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_sum_33.run(*args, 224, 23699, grid=grid(224), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_sum_33.benchmark_all_configs(*args, 224, 23699, grid=grid(224))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
