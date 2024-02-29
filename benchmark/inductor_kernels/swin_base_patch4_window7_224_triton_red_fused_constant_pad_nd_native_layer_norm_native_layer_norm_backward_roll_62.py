
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_constant_pad_nd_native_layer_norm_native_layer_norm_backward_roll_62', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_constant_pad_nd_native_layer_norm_native_layer_norm_backward_roll_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (512*(((11 + ((r2 + (128*x0)) % 14)) % 14) % 7)) + (3584*(((11 + (((r2 + (128*x0)) // 14) % 14)) % 14) % 7)) + (25088*(((11 + ((r2 + (128*x0)) % 14)) % 14) // 7)) + (50176*(((11 + (((r2 + (128*x0)) // 14) % 14)) % 14) // 7)) + (100352*((r2 + (128*x0)) // 196))), rmask & xmask, other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((12544, 512), (512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 14, 14, 512), (100352, 7168, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((64, 14, 14, 1), (196, 14, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 14, 14, 1), (196, 14, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((512, 98), (98, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((512, 98), (98, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_constant_pad_nd_native_layer_norm_native_layer_norm_backward_roll_62.run(*args, 50176, 128, grid=grid(50176), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_constant_pad_nd_native_layer_norm_native_layer_norm_backward_roll_62.benchmark_all_configs(*args, 50176, 128, grid=grid(50176))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
