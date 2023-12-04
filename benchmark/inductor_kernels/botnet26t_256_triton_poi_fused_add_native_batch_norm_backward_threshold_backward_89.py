
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

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_89', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]})
@triton.jit
def triton_poi_fused_add_native_batch_norm_backward_threshold_backward_89(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp12, None)


def get_args():
    arg_0 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((128, 256, 64, 64), (1048576, 1, 16384, 256), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_89.run(*args, 134217728, grid=grid(134217728), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_add_native_batch_norm_backward_threshold_backward_89.benchmark_all_configs(*args, 134217728, grid=grid(134217728))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
