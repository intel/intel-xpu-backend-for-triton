
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

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_74', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]})
@triton.jit
def triton_poi_fused_col2im_74(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21676032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x7 = (xindex // 8064) % 42
    x9 = (xindex // 192) % 42
    x0 = xindex % 192
    x1 = (xindex // 192) % 14
    x2 = (xindex // 2688) % 3
    x3 = (xindex // 8064) % 14
    x4 = (xindex // 112896) % 3
    x5 = (xindex // 338688)
    tmp0 = tl.load(in_ptr0 + (x7), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x9), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (96*x4) + (288*x1) + (4032*x3) + (56448*((x2 + (3*x4) + (9*x0)) // 288)) + (338688*x5) + (((x2 + (3*x4) + (9*x0)) // 9) % 32)), None).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30, tmp0)
    tmp3 = tl.where(tmp2 < 0, tmp2 + 30, tmp2)
    tmp5 = tmp4.to(tl.float32)
    tl.atomic_add(out_ptr0 + (tmp3 + (30*tmp1) + (900*x0) + (172800*x5)), tmp5, None)


def get_args():
    arg_0 = rand_strided((3, 14, 1, 1), (14, 1, 1, 1), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((3, 14), (14, 1), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((75264, 9, 32), (288, 32, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((64, 192, 30, 30), (172800, 900, 30, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_col2im_74.run(*args, 21676032, grid=grid(21676032), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_col2im_74.benchmark_all_configs(*args, 21676032, grid=grid(21676032))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
