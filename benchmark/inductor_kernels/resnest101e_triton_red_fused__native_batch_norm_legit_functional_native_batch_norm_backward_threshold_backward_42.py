
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
    size_hints=[512, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_42', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*(x0 % 256)) + (65536*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + ((256*r2) + (x0 % 256)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr4 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 256.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp11 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp21, xmask)
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tmp21 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((64, 512, 16, 16), (131072, 256, 16, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 256, 16, 16), (65536, 256, 16, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((64, 2, 1, 256), (512, 256, 256, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((64, 512, 16, 16), (131072, 256, 16, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_42.run(*args, 512, 16384, grid=grid(512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_42.benchmark_all_configs(*args, 512, 16384, grid=grid(512))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
