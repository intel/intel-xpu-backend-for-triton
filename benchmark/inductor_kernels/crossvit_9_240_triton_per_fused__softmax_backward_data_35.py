
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
    size_hints=[262144, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_35(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 205312
    XBLOCK: tl.constexpr = 1
    rnumel = 401
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (401*x0)), rmask & xmask, other=0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (401*x0)), rmask & xmask, other=0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tmp3 * tmp8
    tmp10 = tmp4 - tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (401*x0)), tmp11, rmask & xmask)


def get_args():
    arg_0 = rand_strided((512, 401, 401), (160801, 401, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 4, 401, 401), (643204, 160801, 401, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 4, 401, 401), (643204, 160801, 401, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax_backward_data_35.run(*args, 205312, 401, grid=grid(205312), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax_backward_data_35.benchmark_all_configs(*args, 205312, 401, grid=grid(205312))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
