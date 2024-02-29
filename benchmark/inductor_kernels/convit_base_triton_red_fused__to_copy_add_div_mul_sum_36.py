
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
    size_hints=[256, 262144],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_sum_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_sum_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*(((r2 + (175616*x1)) // 196) % 196)) + (38416*x0) + (614656*((r2 + (175616*x1)) // 38416)) + (r2 % 196)), rmask & xmask, other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + ((196*x0) + (3136*((r2 + (175616*x1)) // 38416)) + (((r2 + (175616*x1)) // 196) % 196)), rmask & xmask, other=0)
        tmp4 = tl.load(in_ptr2 + ((196*x0) + (3136*((r2 + (175616*x1)) // 38416)) + (((r2 + (175616*x1)) // 196) % 196)), rmask & xmask, other=0)
        tmp6 = tl.load(in_ptr3 + ((196*(((r2 + (175616*x1)) // 196) % 196)) + (38416*x0) + (614656*((r2 + (175616*x1)) // 38416)) + (r2 % 196)), rmask & xmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + ((196*(((r2 + (175616*x1)) // 196) % 196)) + (38416*x0) + (614656*((r2 + (175616*x1)) // 38416)) + (r2 % 196)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 / tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp5 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((1024, 196, 196), (38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 16, 196, 1), (3136, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 16, 196, 1), (3136, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 16, 196, 196), (614656, 38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((64, 16, 196, 196), (614656, 38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((1, 16, 1, 1, 14), (224, 1, 224, 224, 16), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((1, 16, 1, 1, 14), (224, 1, 224, 224, 16), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_div_mul_sum_36.run(*args, 224, 175616, grid=grid(224), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_div_mul_sum_36.benchmark_all_configs(*args, 224, 175616, grid=grid(224))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
