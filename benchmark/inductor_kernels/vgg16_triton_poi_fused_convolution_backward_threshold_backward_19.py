
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


def get_args():
    arg_0 = rand_strided((64, 64, 224, 224), (3211264, 1, 14336, 64), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 64, 224, 224), (3211264, 1, 14336, 64), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_19.run(*args, 205520896, grid=grid(205520896), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_convolution_backward_threshold_backward_19.benchmark_all_configs(*args, 205520896, grid=grid(205520896))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
