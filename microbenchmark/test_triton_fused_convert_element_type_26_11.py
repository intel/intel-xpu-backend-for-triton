
from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._dynamo.testing import rand_strided
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = XPUAsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream


# Key point: large tensor load/store
triton_fused_convert_element_type_26_11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]})
@triton.jit
def triton_fused_convert_element_type_26_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_8, view, addmm, convert_element_type_4, getitem_1, rsqrt, view_2, sub_2, convert_element_type_10, permute_2, permute_6, tangents_1, tangents_2 = args
    args.clear()
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0) # no-op to ensure context
        stream0 = get_xpu_stream(0)
        buf4 = rand_strided((16384, 768), (768, 1), device='xpu', dtype=torch.float16)
        buf19 = buf4; del buf4  # reuse
        buf25 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu', dtype=torch.float32)
        triton_fused_convert_element_type_26_11.run(buf19, buf25, 12582912, grid=grid(12582912), stream=stream0)
        return (buf25, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from intel_extension_for_pytorch._inductor.xpu.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='xpu:0', dtype=torch.int64)
    view = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_4 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float16)
    getitem_1 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    rsqrt = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    view_2 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    sub_2 = rand_strided((16384, 30522), (30522, 1), device='xpu:0', dtype=torch.float32)
    convert_element_type_10 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    permute_2 = rand_strided((30522, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_6 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    tangents_1 = rand_strided((), (), device='xpu:0', dtype=torch.float32)
    tangents_2 = rand_strided((16384, 30522), (30522, 1), device='xpu:0', dtype=torch.float16)
    print_performance(lambda: call([primals_3, primals_8, view, addmm, convert_element_type_4, getitem_1, rsqrt, view_2, sub_2, convert_element_type_10, permute_2, permute_6, tangents_1, tangents_2]))
