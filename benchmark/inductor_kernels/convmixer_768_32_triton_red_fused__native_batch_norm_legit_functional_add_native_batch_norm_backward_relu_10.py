
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_relu_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (393216*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (393216*x1)), rmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (x0 + (768*r2) + (393216*x1)), rmask, other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tmp8 = triton_helpers.maximum(0, tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)


def get_args():
    arg_0 = rand_strided((32, 768, 32, 32), (786432, 1, 24576, 768), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 768, 32, 32), (786432, 1, 24576, 768), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((32, 768, 32, 32), (786432, 1, 24576, 768), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((768, 64), (1, 768), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((768, 64), (1, 768), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_relu_10.run(*args, 49152, 512, grid=grid(49152), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_relu_10.benchmark_all_configs(*args, 49152, 512, grid=grid(49152))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
