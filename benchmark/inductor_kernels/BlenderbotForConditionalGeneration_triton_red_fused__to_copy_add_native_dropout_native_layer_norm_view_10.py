
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
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*bf16', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp12 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (2560*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.1
        tmp5 = tmp3 > tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = 1.1111111111111112
        tmp10 = tmp8 * tmp9
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = tmp10.to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask, tmp18_weight_next, tmp18_weight)
        tl.store(out_ptr1 + (r1 + (2560*x0)), tmp5, rmask)
        tl.store(in_out_ptr0 + (r1 + (2560*x0)), tmp10, rmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp18, None)
    tmp29_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp22 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp25 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp29_mean_next, tmp29_m2_next, tmp29_weight_next = triton_helpers.welford_reduce(
            tmp28, tmp29_mean, tmp29_m2, tmp29_weight,
        )
        tmp29_mean = tl.where(rmask, tmp29_mean_next, tmp29_mean)
        tmp29_m2 = tl.where(rmask, tmp29_m2_next, tmp29_m2)
        tmp29_weight = tl.where(rmask, tmp29_weight_next, tmp29_weight)
    tmp29_tmp, tmp30_tmp, tmp31_tmp = triton_helpers.welford(
        tmp29_mean, tmp29_m2, tmp29_weight, 1
    )
    tmp29 = tmp29_tmp[:, None]
    tmp30 = tmp30_tmp[:, None]
    tmp31 = tmp31_tmp[:, None]
    tmp32 = 2560.0
    tmp33 = tmp30 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp36, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp37 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask, other=0)
        tmp38 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask, other=0).to(tl.float32)
        tmp41 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask, other=0).to(tl.float32)
        tmp46 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp48 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp37 + tmp39
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp40 + tmp42
        tmp44 = tmp43 - tmp18
        tmp45 = tmp44 * tmp36
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tmp50 = tmp49.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (2560*x0)), tmp50, rmask)


def get_args():
    arg_0 = rand_strided((16, 128, 2560), (327680, 2560, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((3,), (1,), device='xpu:0', dtype=torch.int64)
    arg_3 = rand_strided((16, 128, 2560), (327680, 2560, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((16, 128, 2560), (327680, 2560, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((2560,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((2560,), (1,), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((16, 128, 2560), (327680, 2560, 1), device='xpu:0', dtype=torch.bool)
    arg_8 = rand_strided((16, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((2048, 2560), (2560, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_10 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_10.run(*args, 2048, 2560, grid=grid(2048), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_10.benchmark_all_configs(*args, 2048, 2560, grid=grid(2048))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=2) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
