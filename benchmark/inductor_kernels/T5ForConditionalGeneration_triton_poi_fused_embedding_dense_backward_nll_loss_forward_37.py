
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_nll_loss_forward_37(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1048576
    x2 = xindex
    x1 = (xindex // 1048576)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (8388608 + x2), None)
    tmp5 = tl.load(in_ptr1 + (16777216 + x2), None)
    tmp7 = tl.load(in_ptr1 + (25165824 + x2), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32, tmp0)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], False, tl.int1)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp8)
    tl.atomic_add(out_ptr0 + (x1 + (8*tmp1)), tmp11, None)


def get_args():
    arg_0 = rand_strided((1024, 1024), (1024, 1), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 8), (8, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_37.run(*args, 8388608, grid=grid(8388608), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_embedding_dense_backward_nll_loss_forward_37.benchmark_all_configs(*args, 8388608, grid=grid(8388608))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
