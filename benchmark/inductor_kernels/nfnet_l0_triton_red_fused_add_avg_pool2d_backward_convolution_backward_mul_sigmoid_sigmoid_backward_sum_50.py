
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
    size_hints=[32768, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_convolution_backward_mul_sigmoid_sigmoid_backward_sum_50', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_avg_pool2d_backward_convolution_backward_mul_sigmoid_sigmoid_backward_sum_50(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 56
        r2 = (rindex // 56)
        tmp0 = tl.load(in_out_ptr0 + (r3 + (3136*x0)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2))))) >= 0, 0, 28))) + (784*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) >= 0, 0, 28))), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r3 + (3136*x0)), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r3 + (3136*x0)), rmask, other=0).to(tl.float32)
        tmp2 = tmp1 / 4
        tmp3 = tl.math.max(0, (r2 // 2))
        tmp4 = tl.math.min(28, 1 + (r2 // 2))
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (r1 // 2))
        tmp7 = tl.math.min(28, 1 + (r1 // 2))
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp12 = tmp0 + tmp11
        tmp13 = 0.9805806756909201
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 * tmp15
        tmp17 = 0.2
        tmp18 = tmp16 * tmp17
        tmp19 = 2.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
        tl.store(in_out_ptr0 + (r3 + (3136*x0)), tmp16, rmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp27 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tmp24.to(tl.float32)
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.0
    tmp31 = tmp30 - tmp29
    tmp32 = tmp29 * tmp31
    tmp33 = tmp26 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, None)


def get_args():
    arg_0 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_avg_pool2d_backward_convolution_backward_mul_sigmoid_sigmoid_backward_sum_50.run(*args, 32768, 3136, grid=grid(32768), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_avg_pool2d_backward_convolution_backward_mul_sigmoid_sigmoid_backward_sum_50.benchmark_all_configs(*args, 32768, 3136, grid=grid(32768))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=2) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
