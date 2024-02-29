
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@persistent_reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 98304
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.3895313892515355e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.load(in_ptr1 + load_seed_offset)
    tmp16 = r1 + (512*x0)
    tmp17 = tl.rand(tmp15, (tmp16).to(tl.uint32))
    tmp18 = 0.9
    tmp19 = tmp17 < tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.0
    tmp22 = tmp21 - tmp20
    tmp23 = (tmp22 != 0)
    tmp24 = tmp10 / tmp14
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 0.0
    tmp27 = tl.where(tmp1, tmp26, tmp25)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = 1.1111111111111112
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp23, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp27, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp30, rmask)


def get_args():
    arg_0 = rand_strided((192, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((37,), (1,), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bool)
    arg_3 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5.run(*args, 98304, 512, grid=grid(98304), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5.benchmark_all_configs(*args, 98304, 512, grid=grid(98304))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
