
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
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_add_index_mul_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_add_index_mul_23(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tl.where(tmp4 < 0, tmp4 + 196, tmp4)
        # tl.device_assert(((0 <= tmp5) & (tmp5 < 196)) | ~rmask, "index out of bounds: 0 <= tmp5 < 196")
        tmp6 = tl.load(in_ptr2 + (tmp5 + (196*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tmp3 + tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp12 = 0.25
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tl.where(tmp15 < 0, tmp15 + 196, tmp15)
        # tl.device_assert(((0 <= tmp16) & (tmp16 < 196)) | ~rmask, "index out of bounds: 0 <= tmp16 < 196")
        tmp17 = tl.load(in_ptr2 + (tmp16 + (196*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp18 = tmp14 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp24 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp25 = 0.25
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tl.where(tmp28 < 0, tmp28 + 196, tmp28)
        # tl.device_assert(((0 <= tmp29) & (tmp29 < 196)) | ~rmask, "index out of bounds: 0 <= tmp29 < 196")
        tmp30 = tl.load(in_ptr2 + (tmp29 + (196*x1)), rmask, other=0)
        tmp31 = tmp27 + tmp30
        tmp32 = tmp31 - tmp9
        tmp33 = tl.exp(tmp32)
        tmp34 = tmp33 / tmp22
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (196*x4)), tmp34, rmask)
        tl.store(out_ptr3 + (r3 + (196*x4)), tmp35, rmask)


def get_args():
    arg_0 = rand_strided((512, 196, 196), (38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((196, 196), (196, 1), device='xpu:0', dtype=torch.int64)
    arg_2 = rand_strided((4, 196), (196, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 4, 196, 196), (153664, 38416, 196, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((128, 4, 196, 196), (153664, 38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax__to_copy_add_index_mul_23.run(*args, 100352, 196, grid=grid(100352), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax__to_copy_add_index_mul_23.benchmark_all_configs(*args, 100352, 196, grid=grid(100352))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
