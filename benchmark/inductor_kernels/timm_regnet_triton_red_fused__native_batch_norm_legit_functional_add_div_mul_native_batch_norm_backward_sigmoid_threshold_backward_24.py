
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
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (896*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (x0 + (896*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp12 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr2 + (x0), tmp25, xmask)


def get_args():
    arg_0 = rand_strided((32, 896, 14, 14), (175616, 196, 14, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 896, 14, 14), (175616, 196, 14, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((32, 896, 1, 1), (896, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((32, 896, 1, 1), (896, 1, 1, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((32, 896, 14, 14), (175616, 196, 14, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((1, 896, 1, 1), (896, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((896,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((896,), (1,), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((896,), (1,), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((896,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_24.run(*args, 896, 6272, grid=grid(896), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_24.benchmark_all_configs(*args, 896, 6272, grid=grid(896))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
