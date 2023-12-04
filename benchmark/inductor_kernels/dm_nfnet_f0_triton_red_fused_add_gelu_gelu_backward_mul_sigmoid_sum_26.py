
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
    size_hints=[256, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
)
@triton.jit
def triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp48 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (131072*x0)), rmask & xmask, other=0)
        tmp4 = tl.load(in_ptr1 + (r1 + (131072*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (((r1 + (131072*x0)) // 144)), rmask & xmask, other=0).to(tl.float32)
        tmp14 = tl.load(in_ptr3 + (r1 + (131072*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp20 = tl.load(in_ptr4 + (r1 + (131072*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp41 = tl.load(in_ptr5 + (r1 + (131072*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr6 + (((r1 + (131072*x0)) // 144)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.2
        tmp3 = tmp1 * tmp2
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tmp4 * tmp6
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = 0.9128709291752768
        tmp16 = tmp14 * tmp15
        tmp17 = 1.7015043497085571
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 0.7071067811865476
        tmp23 = tmp21 * tmp22
        tmp24 = tl.math.erf(tmp23)
        tmp25 = 1.0
        tmp26 = tmp24 + tmp25
        tmp27 = 0.5
        tmp28 = tmp26 * tmp27
        tmp29 = tmp21 * tmp21
        tmp30 = -0.5
        tmp31 = tmp29 * tmp30
        tmp32 = tl.exp(tmp31)
        tmp33 = 0.3989422804014327
        tmp34 = tmp32 * tmp33
        tmp35 = tmp21 * tmp34
        tmp36 = tmp28 + tmp35
        tmp37 = tmp19 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp1 + tmp38
        tmp40 = tmp39 * tmp2
        tmp43 = tl.sigmoid(tmp42)
        tmp44 = tmp41 * tmp43
        tmp45 = tmp44 * tmp8
        tmp46 = tmp40 * tmp45
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
        tmp49 = _tmp48 + tmp47
        _tmp48 = tl.where(rmask & xmask, tmp49, _tmp48)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp48, xmask)


def get_args():
    arg_0 = rand_strided((128, 1536, 12, 12), (221184, 144, 12, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 1536, 12, 12), (221184, 144, 12, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 1536, 1, 1), (1536, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 1536, 12, 12), (221184, 144, 12, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 1536, 12, 12), (221184, 144, 12, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((128, 1536, 12, 12), (221184, 144, 12, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((128, 1536, 1, 1), (1536, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((216,), (1,), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((216,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_26.run(*args, 216, 131072, grid=grid(216), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_26.benchmark_all_configs(*args, 216, 131072, grid=grid(216))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
