
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr2 + (x2 + (196*r1) + (75264*x3)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tmp3 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp20 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp26 = tl.load(in_ptr2 + (x2 + (196*r1) + (75264*x3)), rmask & xmask, other=0).to(tl.float32)
        tmp18 = 384.0
        tmp19 = tmp11 / tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23 * tmp18
        tmp25 = tmp24 - tmp5
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp9
        tmp29 = tmp28 * tmp11
        tmp30 = tmp29 * tmp15
        tmp31 = tmp25 - tmp30
        tmp32 = tmp19 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp17 + tmp33
        tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp34, rmask & xmask)


def get_args():
    arg_0 = rand_strided((128, 196, 384), (75264, 384, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((25088, 384), (384, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((384,), (1,), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 196, 384), (75264, 1, 196), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 196, 1), (196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 196, 1), (196, 1, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_native_layer_norm_backward_13.run(*args, 25088, 384, grid=grid(25088), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_native_layer_norm_backward_13.benchmark_all_configs(*args, 25088, 384, grid=grid(25088))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
