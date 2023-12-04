
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_eq_ge_masked_fill_mul_neg_scalar_tensor_sum_where_56', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused_add_div_eq_ge_masked_fill_mul_neg_scalar_tensor_sum_where_56(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x2 = (xindex // 768)
    x5 = xindex % 768
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    tmp3 = tl.load(in_ptr2 + (x1 + (16*x0) + (768*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (784*x4)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x5 + (2304*r3) + (1806336*x2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = -tmp0
        tmp4 = 1e-12
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = tmp2 / tmp5
        tmp7 = tmp6 / tmp5
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp12 = tl.load(in_ptr0 + (r3 + (784*x4)), rmask & xmask, other=0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (x5 + (2304*r3) + (1806336*x2)), rmask & xmask, other=0).to(tl.float32)
        tmp13 = 1e-12
        tmp14 = triton_helpers.maximum(tmp3, tmp13)
        tmp15 = tmp12 / tmp14
        tmp16 = tmp3 >= tmp13
        tmp17 = 0.0
        tmp18 = tl.where(tmp16, tmp10, tmp17)
        tmp19 = tmp3 == tmp17
        tmp21 = tmp20 / tmp3
        tmp22 = tl.where(tmp19, tmp17, tmp21)
        tmp23 = tmp18 * tmp22
        tmp24 = tmp15 + tmp23
        tl.store(out_ptr1 + (r3 + (784*x4)), tmp24, rmask & xmask)


def get_args():
    arg_0 = rand_strided((80, 48, 784), (37632, 784, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((5, 16, 48, 784), (1806336, 48, 1, 2304), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((5, 16, 48, 1), (768, 1, 16, 16), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((5, 16, 48, 784), (602112, 37632, 784, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_div_eq_ge_masked_fill_mul_neg_scalar_tensor_sum_where_56.run(*args, 3840, 784, grid=grid(3840), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_div_eq_ge_masked_fill_mul_neg_scalar_tensor_sum_where_56.benchmark_all_configs(*args, 3840, 784, grid=grid(3840))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
