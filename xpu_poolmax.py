import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ['TRITON_INTERPRET'] = '0'
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['LLVM_IR_ENABLE_DUMP'] = '1'

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 177020928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 147)
    x2 = ((xindex // 9408) % 147)
    x3 = xindex // 1382976
    x4 = ((xindex // 64) % 21609)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp35 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp51 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 9, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 9), "index out of bounds: 0 <= tmp4 < 9")
    tmp7 = tmp4 + 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp4 // 3) + 294*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 9), "index out of bounds: 0 <= tmp15 < 9")
    tmp18 = tmp15 + 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp15 // 3) + 294*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp19 = tmp18 == tmp8
    tmp20 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))
    tmp21 = ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))
    tmp22 = tmp20 < tmp21
    tmp23 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp24 = ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))
    tmp25 = tmp23 < tmp24
    tmp26 = tmp22 & tmp25
    tmp27 = tmp26 & tmp19
    tmp28 = tmp11 + tmp17
    tmp29 = tl.where(tmp27, tmp28, tmp11)
    tmp31 = tmp30 + tmp1
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert((0 <= tmp33) & (tmp33 < 9), "index out of bounds: 0 <= tmp33 < 9")
    tmp36 = tmp33 + 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp33 // 3) + 294*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp37 = tmp36 == tmp8
    tmp38 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))
    tmp39 = tmp38 < tmp21
    tmp40 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp41 = tmp40 < tmp24
    tmp42 = tmp39 & tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tmp29 + tmp35
    tmp45 = tl.where(tmp43, tmp44, tmp29)
    tmp47 = tmp46 + tmp1
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert((0 <= tmp49) & (tmp49 < 9), "index out of bounds: 0 <= tmp49 < 9")
    tmp52 = tmp49 + 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp49 // 3) + 294*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp53 = tmp52 == tmp8
    tmp54 = tmp39 & tmp25
    tmp55 = tmp54 & tmp53
    tmp56 = tmp45 + tmp51
    tmp57 = tl.where(tmp55, tmp56, tmp45)
    tl.store(out_ptr0 + (x7), tmp57, None)


def get_args():
    arg_0 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.int8)
    arg_1 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.float16)
    arg_2 = rand_strided((128, 64, 147, 147), (1382976, 1, 9408, 64), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 177020928


def benchmark_kernel(xblock, num_stages, num_warps, num_ctas):
    """Benchmark a single configuration"""
    args = get_args()
    
    def call(args):
        with torch.xpu._DeviceGuard(0):
            torch.xpu.set_device(0)
            stream0 = get_raw_stream(0)
            
            grid = lambda meta: (triton.cdiv(args[3], meta['XBLOCK']),)
            triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139[grid](
                *args, XBLOCK=xblock, num_stages=num_stages, num_warps=num_warps, num_ctas=num_ctas
            )
    
    from torch._inductor.runtime.benchmarking import benchmarker
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    return ms


def main():    
    xblock_sizes = [32, 64, 128, 256, 512, 1024]
    num_stages_options = [1, 2, 3, 4, 5]
    num_warps_options = [1, 2, 4, 8]
    num_ctas_options = [1]
    #num_ctas_options = [1, 2, 4, 8]

    num_gb = 0.48500736
    
    for xblock in xblock_sizes:
        for num_stages in num_stages_options:
            for num_warps in num_warps_options:
                for num_ctas in num_ctas_options:
                    ms = benchmark_kernel(xblock, num_stages, num_warps, num_ctas)
                    gb_per_s = num_gb / (ms / 1e3)
                    print(f"xblock={xblock}    num_stages={num_stages}    num_warps={num_warps}    num_ctas={num_ctas}    {ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


if __name__ == '__main__':
    main()import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ['TRITON_INTERPRET'] = '0'
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['LLVM_IR_ENABLE_DUMP'] = '1'

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 177020928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 147)
    x2 = ((xindex // 9408) % 147)
    x3 = xindex // 1382976
    x4 = ((xindex // 64) % 21609)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp35 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None)
    tmp51 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 4672*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))))) + 341056*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 9, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 9), "index out of bounds: 0 <= tmp4 < 9")
    tmp7 = tmp4 + 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp4 // 3) + 294*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 9), "index out of bounds: 0 <= tmp15 < 9")
    tmp18 = tmp15 + 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp15 // 3) + 294*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp19 = tmp18 == tmp8
    tmp20 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))
    tmp21 = ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))
    tmp22 = tmp20 < tmp21
    tmp23 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp24 = ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))
    tmp25 = tmp23 < tmp24
    tmp26 = tmp22 & tmp25
    tmp27 = tmp26 & tmp19
    tmp28 = tmp11 + tmp17
    tmp29 = tl.where(tmp27, tmp28, tmp11)
    tmp31 = tmp30 + tmp1
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert((0 <= tmp33) & (tmp33 < 9), "index out of bounds: 0 <= tmp33 < 9")
    tmp36 = tmp33 + 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp33 // 3) + 294*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp37 = tmp36 == tmp8
    tmp38 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))
    tmp39 = tmp38 < tmp21
    tmp40 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp41 = tmp40 < tmp24
    tmp42 = tmp39 & tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tmp29 + tmp35
    tmp45 = tl.where(tmp43, tmp44, tmp29)
    tmp47 = tmp46 + tmp1
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert((0 <= tmp49) & (tmp49 < 9), "index out of bounds: 0 <= tmp49 < 9")
    tmp52 = tmp49 + 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + 144*(tmp49 // 3) + 294*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0)))) <= ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73))))) + ((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) * (((-1) + ((73) * ((73) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (73)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x2,  2))) + (triton_helpers.div_floor_integer((-1) + x2,  2)) * ((triton_helpers.div_floor_integer((-1) + x2,  2)) > (0))))))
    tmp53 = tmp52 == tmp8
    tmp54 = tmp39 & tmp25
    tmp55 = tmp54 & tmp53
    tmp56 = tmp45 + tmp51
    tmp57 = tl.where(tmp55, tmp56, tmp45)
    tl.store(out_ptr0 + (x7), tmp57, None)


def get_args():
    arg_0 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.int8)
    arg_1 = rand_strided((128, 64, 73, 73), (341056, 1, 4672, 64), device='xpu:0', dtype=torch.float16)
    arg_2 = rand_strided((128, 64, 147, 147), (1382976, 1, 9408, 64), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 177020928


def benchmark_kernel(xblock, num_stages, num_warps, num_ctas):
    """Benchmark a single configuration"""
    args = get_args()
    
    def call(args):
        with torch.xpu._DeviceGuard(0):
            torch.xpu.set_device(0)
            stream0 = get_raw_stream(0)
            
            grid = lambda meta: (triton.cdiv(args[3], meta['XBLOCK']),)
            triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_139[grid](
                *args, XBLOCK=xblock, num_stages=num_stages, num_warps=num_warps, num_ctas=num_ctas
            )
    
    from torch._inductor.runtime.benchmarking import benchmarker
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    return ms


def main():    
    xblock_sizes = [32, 64, 128, 256, 512, 1024]
    num_stages_options = [1, 2, 3, 4, 5]
    num_warps_options = [1, 2, 4, 8]
    num_ctas_options = [1]
    #num_ctas_options = [1, 2, 4, 8]

    num_gb = 0.48500736
    
    for xblock in xblock_sizes:
        for num_stages in num_stages_options:
            for num_warps in num_warps_options:
                for num_ctas in num_ctas_options:
                    ms = benchmark_kernel(xblock, num_stages, num_warps, num_ctas)
                    gb_per_s = num_gb / (ms / 1e3)
                    print(f"xblock={xblock}    num_stages={num_stages}    num_warps={num_warps}    num_ctas={num_ctas}    {ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


if __name__ == '__main__':
    main()
