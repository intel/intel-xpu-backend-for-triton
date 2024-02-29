
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
    size_hints=[131072, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_constant_pad_nd_mul_native_batch_norm_backward_164', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_constant_pad_nd_mul_native_batch_norm_backward_164(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr1 + (x0 + (96*r2) + (196608*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (x0 + (96*r2) + (196608*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp0 = ((r2 + (2048*x1)) // 112) % 112
        tmp1 = tl.full([1, 1], 113, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (2048*x1)) % 112
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (96*((r2 + (2048*x1)) % 112)) + (10848*(((r2 + (2048*x1)) // 112) % 112)) + (1225824*((r2 + (2048*x1)) // 12544))), rmask & tmp5 & xmask, other=0).to(tl.float32)
        tmp7 = tl.where(tmp5, tmp6, 0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)


def get_args():
    arg_0 = rand_strided((128, 96, 113, 113), (1225824, 1, 10848, 96), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 96, 112, 112), (1204224, 1, 10752, 96), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 96, 112, 112), (1204224, 1, 10752, 96), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((96, 784), (1, 96), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((96, 784), (1, 96), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_constant_pad_nd_mul_native_batch_norm_backward_164.run(*args, 75264, 2048, grid=grid(75264), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__native_batch_norm_legit_functional_constant_pad_nd_mul_native_batch_norm_backward_164.benchmark_all_configs(*args, 75264, 2048, grid=grid(75264))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
