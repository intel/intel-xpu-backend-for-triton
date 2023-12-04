
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_div_full_native_dropout_where_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_div_full_native_dropout_where_6(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = 8.0
        tmp3 = tmp1 / tmp2
        tmp4 = -3.3895313892515355e+38
        tmp5 = tl.where(tmp0, tmp3, tmp4)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = -3.3895313892515355e+38
        tmp15 = tl.where(tmp10, tmp13, tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr2 + load_seed_offset)
        tmp23 = r2 + (512*x3)
        tmp24 = tl.rand(tmp22, (tmp23).to(tl.uint32))
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp24, rmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr2 + (r2 + (512*x3)), rmask, other=0)
        tmp29 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp30 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, other=0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = -3.3895313892515355e+38
        tmp34 = tl.where(tmp29, tmp32, tmp33)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp8
        tmp37 = tl.exp(tmp36)
        tmp38 = tmp37 / tmp20
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp28.to(tl.float32)
        tmp41 = tmp40 * tmp39
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(out_ptr3 + (r2 + (512*x3)), tmp28, rmask)
        tl.store(out_ptr4 + (r2 + (512*x3)), tmp39, rmask)
        tl.store(out_ptr5 + (r2 + (512*x3)), tmp43, rmask)


def get_args():
    arg_0 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((48, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((37,), (1,), device='xpu:0', dtype=torch.int64)
    arg_3 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bool)
    arg_5 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_7 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax__to_copy_div_full_native_dropout_where_6.run(*args, 24576, 512, grid=grid(24576), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax__to_copy_div_full_native_dropout_where_6.benchmark_all_configs(*args, 24576, 512, grid=grid(24576))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
