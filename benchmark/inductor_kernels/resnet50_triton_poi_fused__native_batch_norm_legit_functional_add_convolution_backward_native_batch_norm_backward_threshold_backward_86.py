
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

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*bf16', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*bf16', 15: '*bf16', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_86', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x2), None).to(tl.float32)
    tmp28 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 9.964923469387754e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 * tmp13
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp7 - tmp35
    tmp37 = tmp36 - tmp21
    tmp39 = tmp32 * tmp38
    tmp40 = tmp37 * tmp39
    tmp41 = tmp25.to(tl.float32)
    tmp42 = tmp40.to(tl.float32)
    tl.store(out_ptr2 + (x2), tmp41, None)
    tl.store(out_ptr3 + (x2), tmp42, None)


def get_args():
    arg_0 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_11 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_12 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_13 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_14 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_15 = rand_strided((32, 256, 56, 56), (802816, 1, 14336, 256), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_86.run(*args, 25690112, grid=grid(25690112), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_86.benchmark_all_configs(*args, 25690112, grid=grid(25690112))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
