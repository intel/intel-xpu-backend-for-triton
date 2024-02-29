
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

@pointwise(size_hints=[536870912], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i1', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_263', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_263(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 308281344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 6.228077168367346e-07
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp23, None)


def get_args():
    arg_0 = rand_strided((128, 192, 112, 112), (2408448, 1, 21504, 192), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 192, 112, 112), (2408448, 1, 21504, 192), device='xpu:0', dtype=torch.bool)
    arg_2 = rand_strided((128, 192, 112, 112), (2408448, 1, 21504, 192), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((192,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_263.run(*args, 308281344, grid=grid(308281344), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_263.benchmark_all_configs(*args, 308281344, grid=grid(308281344))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
