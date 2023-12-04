
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_native_dropout_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_div_native_dropout_6(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.load(in_ptr1 + load_seed_offset)
    tmp15 = r1 + (512*x0)
    tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
    tmp17 = 0.1
    tmp18 = tmp16 > tmp17
    tmp19 = tmp9 / tmp13
    tmp20 = tmp18.to(tl.float32)
    tmp21 = tmp20 * tmp19
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp18, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp19, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp24, rmask)


def get_args():
    arg_0 = rand_strided((64, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((73,), (1,), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((4, 16, 512, 512), (4194304, 262144, 512, 1), device='xpu:0', dtype=torch.bool)
    arg_3 = rand_strided((4, 16, 512, 512), (4194304, 262144, 512, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((4, 16, 512, 512), (4194304, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax__to_copy_add_div_native_dropout_6.run(*args, 32768, 512, grid=grid(32768), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax__to_copy_add_div_native_dropout_6.benchmark_all_configs(*args, 32768, 512, grid=grid(32768))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
