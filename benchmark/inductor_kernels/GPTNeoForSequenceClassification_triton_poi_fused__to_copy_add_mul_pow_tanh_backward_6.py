
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

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_pow_tanh_backward_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_pow_tanh_backward_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp1 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp6 * tmp10
    tmp12 = 0.7978845608028654
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 0.044715
    tmp16 = tmp13 * tmp15
    tmp17 = tmp2.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = 3.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp14 + tmp22
    tmp24 = tmp7 + tmp9
    tmp25 = tmp1 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp23 + tmp27
    tl.store(in_out_ptr0 + (x0), tmp28, None)


def get_args():
    arg_0 = rand_strided((32, 128, 8192), (1048576, 8192, 1), device='xpu:0', dtype=torch.float16)
    arg_1 = rand_strided((4096, 8192), (8192, 1), device='xpu:0', dtype=torch.float16)
    arg_2 = rand_strided((32, 128, 8192), (1048576, 8192, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__to_copy_add_mul_pow_tanh_backward_6.run(*args, 33554432, grid=grid(33554432), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__to_copy_add_mul_pow_tanh_backward_6.benchmark_all_configs(*args, 33554432, grid=grid(33554432))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
