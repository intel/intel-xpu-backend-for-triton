
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
    size_hints=[128, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (401408*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight,
        )
        tmp3_mean = tl.where(rmask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask & xmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tmp15 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = 25088.0
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = 1.0000398612827361
    tmp12 = tmp7 * tmp11
    tmp13 = 0.1
    tmp14 = tmp12 * tmp13
    tmp16 = 0.9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp3 * tmp13
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 + tmp21
    tl.store(out_ptr2 + (x0), tmp10, xmask)
    tl.store(out_ptr3 + (x0), tmp18, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)


def get_args():
    arg_0 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128,), (1,), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 128, 1, 1), (128, 1, 128, 128), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 128, 1, 1), (128, 1, 128, 128), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((128,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_7.run(*args, 128, 25088, grid=grid(128), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__native_batch_norm_legit_functional_7.benchmark_all_configs(*args, 128, 25088, grid=grid(128))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
