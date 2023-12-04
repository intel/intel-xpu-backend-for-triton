
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_sub_sum_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_sub_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp9 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, other=0)
        tmp10 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp19 = tl.load(in_ptr5 + (x0 + (768*r2) + (98304*x1)), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr6 + (x0 + (768*r2) + (98304*x1)), rmask, other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr7 + (x0 + (768*r2) + (98304*x1)), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp33 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = tmp1 / tmp7
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp23 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
        tmp31 = tmp30 + tmp6
        tmp32 = tmp26 / tmp31
        tmp34 = tmp9 - tmp33
        tmp35 = tmp32 * tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask, tmp38, _tmp37)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp28, None)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp37, None)


def get_args():
    arg_0 = rand_strided((2048, 768), (768, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((2048, 768), (768, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((2048, 768), (768, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((2048, 768), (768, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_10 = rand_strided((1, 1, 768, 16), (12288, 12288, 1, 768), device='xpu:0', dtype=torch.float32)
    arg_11 = rand_strided((1, 1, 768, 16), (12288, 12288, 1, 768), device='xpu:0', dtype=torch.float32)
    arg_12 = rand_strided((1, 1, 768, 16), (12288, 12288, 1, 768), device='xpu:0', dtype=torch.float32)
    arg_13 = rand_strided((1, 1, 768, 16), (12288, 12288, 1, 768), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_div_mul_sub_sum_13.run(*args, 12288, 128, grid=grid(12288), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_div_mul_sub_sum_13.benchmark_all_configs(*args, 12288, 128, grid=grid(12288))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
