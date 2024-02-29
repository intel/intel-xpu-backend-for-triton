
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr2 + (r2 + (1536*x3)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
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
        r2 = rindex
        tmp19 = tl.load(in_ptr0 + (x0 + (196*r2) + (150528*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp25 = tl.load(in_ptr2 + (r2 + (1536*x3)), rmask & xmask, other=0).to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp11 / tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22 * tmp17
        tmp24 = tmp23 - tmp5
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp9
        tmp28 = tmp27 * tmp11
        tmp29 = tmp28 * tmp15
        tmp30 = tmp24 - tmp29
        tmp31 = tmp18 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1536*x3)), tmp32, rmask & xmask)


def get_args():
    arg_0 = rand_strided((98304, 196), (196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((768,), (1,), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 196, 768), (301056, 1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 196, 1), (196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((128, 196, 1), (196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 196, 768), (301056, 1536, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_native_layer_norm_backward_12.run(*args, 25088, 768, grid=grid(25088), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_native_layer_norm_backward_12.benchmark_all_configs(*args, 25088, 768, grid=grid(25088))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
