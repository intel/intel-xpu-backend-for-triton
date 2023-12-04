
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
    size_hints=[524288, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_masked_fill_native_dropout_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
)
@triton.jit
def triton_red_fused__softmax_masked_fill_native_dropout_7(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 393216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = -3.3895313892515355e+38
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp10 = -3.3895313892515355e+38
        tmp11 = tl.where(tmp8, tmp10, tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr2 + load_seed_offset)
        tmp19 = r2 + (128*x3)
        tmp20 = tl.rand(tmp18, (tmp19).to(tl.uint32))
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp20, rmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask, other=0)
        tmp25 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, other=0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = 0.1
        tmp24 = tmp22 > tmp23
        tmp27 = -3.3895313892515355e+38
        tmp28 = tl.where(tmp25, tmp27, tmp26)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp6
        tmp31 = tl.exp(tmp30)
        tmp32 = tmp31 / tmp16
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp24.to(tl.float32)
        tmp35 = tmp34 * tmp33
        tmp36 = 1.1111111111111112
        tmp37 = tmp35 * tmp36
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp24, rmask)
        tl.store(out_ptr4 + (r2 + (128*x3)), tmp33, rmask)
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp37, rmask)


def get_args():
    arg_0 = rand_strided((256, 1, 1, 128), (128, 128, 128, 1), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((3072, 128, 128), (16384, 128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((14,), (1,), device='xpu:0', dtype=torch.int64)
    arg_3 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.bool)
    arg_5 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_7 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax_masked_fill_native_dropout_7.run(*args, 393216, 128, grid=grid(393216), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax_masked_fill_native_dropout_7.benchmark_all_configs(*args, 393216, 128, grid=grid(393216))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
