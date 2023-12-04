
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
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_leaky_relu_backward_native_batch_norm_backward_123', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_red_fused_leaky_relu_backward_native_batch_norm_backward_123(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (262144*x0)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (2048*x0)) // 128) % 128)) + (16384*x1) + (2097152*((r2 + (2048*x0)) // 16384)) + (r2 % 128)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 0.01
        tmp4 = tmp2 * tmp3
        tmp5 = tl.where(tmp0, tmp2, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)


def get_args():
    arg_0 = rand_strided((64, 128, 128, 128), (2097152, 1, 16384, 128), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((64, 128, 128, 128), (2097152, 16384, 128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 512), (512, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_leaky_relu_backward_native_batch_norm_backward_123.run(*args, 65536, 2048, grid=grid(65536), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_leaky_relu_backward_native_batch_norm_backward_123.benchmark_all_configs(*args, 65536, 2048, grid=grid(65536))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
