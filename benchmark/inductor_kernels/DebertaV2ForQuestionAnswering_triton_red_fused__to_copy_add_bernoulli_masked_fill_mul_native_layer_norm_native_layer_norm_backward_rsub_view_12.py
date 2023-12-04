
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: '*bf16', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, other=0)
        tmp16 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (1536*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = 0.9
        tmp4 = tmp2 < tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 1.0
        tmp7 = tmp6 - tmp5
        tmp8 = (tmp7 != 0)
        tmp10 = 0.0
        tmp11 = tl.where(tmp8, tmp10, tmp9)
        tmp12 = 1.1111111111111112
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp14 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp20, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp27_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp25 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight,
        )
        tmp27_mean = tl.where(rmask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(rmask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(rmask & xmask, tmp27_weight_next, tmp27_weight)
    tmp27_tmp, tmp28_tmp, tmp29_tmp = triton_helpers.welford(
        tmp27_mean, tmp27_m2, tmp27_weight, 1
    )
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    tmp29 = tmp29_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, other=0)
        tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp40 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp31 = tmp30 - tmp22
        tmp32 = 1536.0
        tmp33 = tmp28 / tmp32
        tmp34 = 1e-07
        tmp35 = tmp33 + tmp34
        tmp36 = tl.math.rsqrt(tmp35)
        tmp37 = tmp31 * tmp36
        tmp39 = tmp37 * tmp38
        tmp41 = tmp39 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp37, rmask & xmask)
        tl.store(out_ptr6 + (r1 + (1536*x0)), tmp42, rmask & xmask)
    tmp43 = 1536.0
    tmp44 = tmp28 / tmp43
    tmp45 = 1e-07
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp47 / tmp43
    tl.store(out_ptr7 + (x0), tmp48, xmask)


def get_args():
    arg_0 = rand_strided((73,), (1,), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((512, 1536), (1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((1, 512, 1536), (786432, 1536, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1536,), (1,), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((1536,), (1,), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((1536,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((1536,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((1, 512, 1536), (786432, 1536, 1), device='xpu:0', dtype=torch.bool)
    arg_8 = rand_strided((1, 512, 1536), (786432, 1536, 1), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((1, 512, 1536), (786432, 1536, 1), device='xpu:0', dtype=torch.float32)
    arg_10 = rand_strided((512, 1536), (1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_11 = rand_strided((1, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_12 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12.run(*args, 512, 1536, grid=grid(512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12.benchmark_all_configs(*args, 512, 1536, grid=grid(512))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
