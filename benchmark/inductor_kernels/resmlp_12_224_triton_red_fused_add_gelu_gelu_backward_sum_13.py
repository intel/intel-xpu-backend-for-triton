
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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_sum_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_red_fused_add_gelu_gelu_backward_sum_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58368
    rnumel = 661
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x0 = xindex % 1536
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (661*x1)
        tmp1 = tl.full([1, 1], 25088, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*((r2 + (661*x1)) % 25088))), rmask & tmp2 & xmask, other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + (1536*((r2 + (661*x1)) % 25088))), rmask & tmp2 & xmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 0.7071067811865476
        tmp10 = tmp8 * tmp9
        tmp11 = tl.math.erf(tmp10)
        tmp12 = 1.0
        tmp13 = tmp11 + tmp12
        tmp14 = 0.5
        tmp15 = tmp13 * tmp14
        tmp16 = tmp8 * tmp8
        tmp17 = -0.5
        tmp18 = tmp16 * tmp17
        tmp19 = tl.exp(tmp18)
        tmp20 = 0.3989422804014327
        tmp21 = tmp19 * tmp20
        tmp22 = tmp8 * tmp21
        tmp23 = tmp15 + tmp22
        tmp24 = tmp4 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.where(tmp2, tmp25, 0)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)


def get_args():
    arg_0 = rand_strided((25088, 1536), (1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((25088, 1536), (1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((1536,), (1,), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 1, 1536, 38), (58368, 58368, 1, 1536), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_gelu_gelu_backward_sum_13.run(*args, 58368, 661, grid=grid(58368), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_gelu_gelu_backward_sum_13.benchmark_all_configs(*args, 58368, 661, grid=grid(58368))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
