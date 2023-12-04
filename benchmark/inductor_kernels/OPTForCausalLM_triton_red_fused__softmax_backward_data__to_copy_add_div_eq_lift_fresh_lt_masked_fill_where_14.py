
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
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_where_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_where_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    x2 = xindex % 2048
    x4 = (xindex // 24576)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (2048*x2) + (4194304*x4)), rmask, eviction_policy='evict_last', other=0)
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.4028234663852886e+38
        tmp12 = tmp10 < tmp11
        tmp13 = tmp10 == tmp11
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 * tmp16
        tmp18 = tmp16 * tmp5
        tmp19 = tmp17 - tmp18
        tmp20 = 2.0
        tmp21 = tmp19 / tmp20
        tmp22 = tl.where(tmp13, tmp21, tmp19)
        tmp23 = 0.0
        tmp24 = tl.where(tmp12, tmp23, tmp22)
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp25, rmask)


def get_args():
    arg_0 = rand_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((2, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((2, 12, 2048, 2048), (50331648, 4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_where_14.run(*args, 49152, 2048, grid=grid(49152), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_where_14.benchmark_all_configs(*args, 49152, 2048, grid=grid(49152))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
