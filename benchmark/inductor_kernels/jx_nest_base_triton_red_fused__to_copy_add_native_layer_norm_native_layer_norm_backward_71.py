
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_71', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp14 = tl.load(in_ptr5 + (x0 + (512*r1)), rmask & xmask, other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp18 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp2 - tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp27 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp28, xmask)


def get_args():
    arg_0 = rand_strided((6272, 512), (512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 1, 196, 512), (100352, 3211264, 512, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 1, 196, 512), (100352, 100352, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((32, 1, 196, 1), (196, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((32, 1, 196, 1), (196, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((6272, 512), (512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((32, 1, 196, 1), (196, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((32, 1, 196, 1), (196, 196, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_10 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    arg_11 = rand_strided((512,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_71.run(*args, 512, 6272, grid=grid(512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_71.benchmark_all_configs(*args, 512, 6272, grid=grid(512))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
