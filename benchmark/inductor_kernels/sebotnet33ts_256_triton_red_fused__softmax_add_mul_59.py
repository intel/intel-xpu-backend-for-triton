
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
    size_hints=[262144, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_59', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_add_mul_59(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp25 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = 0.1767766952966369
        tmp2 = tmp0 * tmp1
        tmp3 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp4 = tl.full([1, 1], 2048, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp7 = tl.full([1, 1], 63, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp9, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp11 = tl.where(tmp9, tmp10, 0.0)
        tmp12 = tl.where(tmp5, tmp11, 0.0)
        tmp13 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp14 = tmp13 < tmp4
        tmp15 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp17, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tl.where(tmp14, tmp19, 0.0)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = triton_helpers.maximum(_tmp25, tmp24)
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
    tmp25 = triton_helpers.max2(_tmp25, 1)[:, None]
    _tmp54 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp28 = 0.1767766952966369
        tmp29 = tmp27 * tmp28
        tmp30 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp31 = tl.full([1, 1], 2048, tl.int64)
        tmp32 = tmp30 < tmp31
        tmp33 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp34 = tl.full([1, 1], 63, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = tmp35 & tmp32
        tmp37 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp36, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp38 = tl.where(tmp36, tmp37, 0.0)
        tmp39 = tl.where(tmp32, tmp38, 0.0)
        tmp40 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp41 = tmp40 < tmp31
        tmp42 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp43 = tmp42 < tmp34
        tmp44 = tmp43 & tmp41
        tmp45 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp44, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp46 = tl.where(tmp44, tmp45, 0.0)
        tmp47 = tl.where(tmp41, tmp46, 0.0)
        tmp48 = tmp39 + tmp47
        tmp49 = tmp29 + tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50 - tmp25
        tmp52 = tl.exp(tmp51)
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(rmask, tmp55, _tmp54)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp56 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0).to(tl.float32)
        tmp57 = 0.1767766952966369
        tmp58 = tmp56 * tmp57
        tmp59 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp60 = tl.full([1, 1], 2048, tl.int64)
        tmp61 = tmp59 < tmp60
        tmp62 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp63 = tl.full([1, 1], 63, tl.int64)
        tmp64 = tmp62 < tmp63
        tmp65 = tmp64 & tmp61
        tmp66 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp65, other=0).to(tl.float32)
        tmp67 = tl.where(tmp65, tmp66, 0.0)
        tmp68 = tl.where(tmp61, tmp67, 0.0)
        tmp69 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp70 = tmp69 < tmp60
        tmp71 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp72 = tmp71 < tmp63
        tmp73 = tmp72 & tmp70
        tmp74 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp73, other=0).to(tl.float32)
        tmp75 = tl.where(tmp73, tmp74, 0.0)
        tmp76 = tl.where(tmp70, tmp75, 0.0)
        tmp77 = tmp68 + tmp76
        tmp78 = tmp58 + tmp77
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tmp79 - tmp25
        tmp81 = tl.exp(tmp80)
        tmp82 = tmp81 / tmp54
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp83, rmask)


def get_args():
    arg_0 = rand_strided((256, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((262144, 63), (63, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((262144, 63), (63, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((256, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax_add_mul_59.run(*args, 262144, 1024, grid=grid(262144), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax_add_mul_59.benchmark_all_configs(*args, 262144, 1024, grid=grid(262144))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
