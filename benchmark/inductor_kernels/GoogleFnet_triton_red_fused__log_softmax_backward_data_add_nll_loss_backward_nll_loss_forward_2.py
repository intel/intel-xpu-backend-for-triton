
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
    size_hints=[8192, 32768],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (32000*x0)), rmask, other=0)
        tmp16 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, other=0)
        tmp27 = tl.load(in_ptr5 + (r1 + (32000*x0)), rmask, other=0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (32000*x0)), tmp31, rmask)


def get_args():
    arg_0 = rand_strided((8192, 32000), (32000, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 512), (512, 1), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((16, 512, 32000), (16384000, 32000, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((8192, 32000), (32000, 1), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((16, 512, 32000), (16384000, 32000, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(*args, 8192, 32000, grid=grid(8192), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.benchmark_all_configs(*args, 8192, 32000, grid=grid(8192))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
