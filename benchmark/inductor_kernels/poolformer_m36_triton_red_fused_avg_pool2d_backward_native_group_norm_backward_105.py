
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
    size_hints=[8192, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_backward_native_group_norm_backward_105', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused_avg_pool2d_backward_native_group_norm_backward_105(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp74 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 56
        r2 = (rindex // 56)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp12 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp20 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp28 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp36 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp42 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp48 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp56 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp62 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask, eviction_policy='evict_last', other=0)
        tmp68 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp71 = tl.load(in_ptr1 + (r3 + (3136*x0)), rmask, other=0)
        tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1)))))
        tmp2 = tmp0 / tmp1
        tmp3 = tl.math.max(0, (-1) + r2)
        tmp4 = tl.math.min(56, 2 + r2)
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (-1) + r1)
        tmp7 = tl.math.min(56, 2 + r1)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp14 = tmp12 / tmp13
        tmp15 = 1 + (tl.math.max(0, (-1) + r1))
        tmp16 = tmp15 < tmp7
        tmp17 = tmp5 & tmp16
        tmp18 = tmp11 + tmp14
        tmp19 = tl.where(tmp17, tmp18, tmp11)
        tmp21 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
        tmp22 = tmp20 / tmp21
        tmp23 = 2 + (tl.math.max(0, (-1) + r1))
        tmp24 = tmp23 < tmp7
        tmp25 = tmp5 & tmp24
        tmp26 = tmp19 + tmp22
        tmp27 = tl.where(tmp25, tmp26, tmp19)
        tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2)))))
        tmp30 = tmp28 / tmp29
        tmp31 = 1 + (tl.math.max(0, (-1) + r2))
        tmp32 = tmp31 < tmp4
        tmp33 = tmp32 & tmp8
        tmp34 = tmp27 + tmp30
        tmp35 = tl.where(tmp33, tmp34, tmp27)
        tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp38 = tmp36 / tmp37
        tmp39 = tmp32 & tmp16
        tmp40 = tmp35 + tmp38
        tmp41 = tl.where(tmp39, tmp40, tmp35)
        tmp43 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
        tmp44 = tmp42 / tmp43
        tmp45 = tmp32 & tmp24
        tmp46 = tmp41 + tmp44
        tmp47 = tl.where(tmp45, tmp46, tmp41)
        tmp49 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
        tmp50 = tmp48 / tmp49
        tmp51 = 2 + (tl.math.max(0, (-1) + r2))
        tmp52 = tmp51 < tmp4
        tmp53 = tmp52 & tmp8
        tmp54 = tmp47 + tmp50
        tmp55 = tl.where(tmp53, tmp54, tmp47)
        tmp57 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
        tmp58 = tmp56 / tmp57
        tmp59 = tmp52 & tmp16
        tmp60 = tmp55 + tmp58
        tmp61 = tl.where(tmp59, tmp60, tmp55)
        tmp63 = 1 + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
        tmp64 = tmp62 / tmp63
        tmp65 = tmp52 & tmp24
        tmp66 = tmp61 + tmp64
        tmp67 = tl.where(tmp65, tmp66, tmp61)
        tmp69 = -tmp68
        tmp70 = tmp69 + tmp67
        tmp72 = tmp70 * tmp71
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
        tmp75 = _tmp74 + tmp73
        _tmp74 = tl.where(rmask, tmp75, _tmp74)
        tl.store(out_ptr0 + (r3 + (3136*x0)), tmp67, rmask)
    tmp74 = tl.sum(_tmp74, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp74, None)
    _tmp81 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp76 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask, other=0)
        tmp78 = tl.load(out_ptr0 + (r3 + (3136*x0)), rmask, other=0)
        tmp77 = -tmp76
        tmp79 = tmp77 + tmp78
        tmp80 = tl.broadcast_to(tmp79, [XBLOCK, RBLOCK])
        tmp82 = _tmp81 + tmp80
        _tmp81 = tl.where(rmask, tmp82, _tmp81)
    tmp81 = tl.sum(_tmp81, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp81, None)


def get_args():
    arg_0 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 96), (96, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((64, 96), (96, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_105.run(*args, 6144, 3136, grid=grid(6144), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_avg_pool2d_backward_native_group_norm_backward_105.benchmark_all_configs(*args, 6144, 3136, grid=grid(6144))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
