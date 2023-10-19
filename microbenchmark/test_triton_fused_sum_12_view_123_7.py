
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


# Key point: reduction_sum
triton_fused_sum_12_view_123_7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_fused_sum_12_view_123_7(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (6291456*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_102, embedding, slice_2, embedding_1, getitem_1, rsqrt, philox_seed_like, view, view_10, convert_element_type_12, view_14, mul_6, view_16, addmm_4, view_18, mul_13, view_20, convert_element_type_35, view_34, mul_17, view_36, addmm_10, view_38, mul_24, view_40, convert_element_type_58, view_54, mul_28, view_56, addmm_16, view_58, mul_35, view_60, convert_element_type_81, view_74, mul_39, view_76, addmm_22, view_78, mul_46, view_80, convert_element_type_104, view_94, mul_50, view_96, addmm_28, view_98, mul_57, view_100, convert_element_type_127, view_114, mul_61, view_116, addmm_34, view_118, mul_68, div_12, permute_66, permute_70, div_13, permute_74, permute_79, permute_80, permute_81, permute_82, permute_85, permute_90, permute_95, div_15, permute_99, permute_103, div_16, permute_107, permute_112, permute_113, permute_114, permute_115, permute_118, permute_123, permute_128, div_18, permute_132, permute_136, div_19, permute_140, permute_145, permute_146, permute_147, permute_148, permute_151, permute_156, permute_161, div_21, permute_165, permute_169, div_22, permute_173, permute_178, permute_179, permute_180, permute_181, permute_184, permute_189, permute_194, div_24, permute_198, permute_202, div_25, permute_206, permute_211, permute_212, permute_213, permute_214, permute_217, permute_222, permute_227, div_27, permute_231, permute_235, div_28, permute_239, permute_244, permute_245, permute_246, permute_247, permute_250, permute_255, permute_260, tangents_1 = args
    args.clear()
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0) # no-op to ensure context
        stream0 = get_xpu_stream(0)
        buf9 = rand_strided((16384, 3072), (3072, 1), device='xpu', dtype=torch.float16)
        buf15 = as_strided(buf9, (128, 128, 3072), (393216, 3072, 1)); del buf9  # reuse
        buf16 = as_strided(buf15, (16384, 3072), (3072, 1)); del buf15  # reuse
        buf19 = rand_strided((1, 3072, 8), (24576, 1, 3072), device='xpu', dtype=torch.float32)
        triton_fused_sum_12_view_123_7.run(buf16, buf19, 24576, 2048, grid=grid(24576), stream=stream0)
        buf75 = buf16; del buf16  # reuse
        buf81 = as_strided(buf75, (128, 128, 3072), (393216, 3072, 1)); del buf75  # reuse
        buf82 = as_strided(buf81, (16384, 3072), (3072, 1)); del buf81  # reuse
        buf85 = buf19; del buf19  # reuse
        triton_fused_sum_12_view_123_7.run(buf82, buf85, 24576, 2048, grid=grid(24576), stream=stream0)
        buf141 = buf82; del buf82  # reuse
        buf147 = as_strided(buf141, (128, 128, 3072), (393216, 3072, 1)); del buf141  # reuse
        buf148 = as_strided(buf147, (16384, 3072), (3072, 1)); del buf147  # reuse
        buf151 = buf85; del buf85  # reuse
        triton_fused_sum_12_view_123_7.run(buf148, buf151, 24576, 2048, grid=grid(24576), stream=stream0)
        buf207 = buf148; del buf148  # reuse
        buf213 = as_strided(buf207, (128, 128, 3072), (393216, 3072, 1)); del buf207  # reuse
        buf214 = as_strided(buf213, (16384, 3072), (3072, 1)); del buf213  # reuse
        buf217 = buf151; del buf151  # reuse
        triton_fused_sum_12_view_123_7.run(buf214, buf217, 24576, 2048, grid=grid(24576), stream=stream0)
        buf273 = buf214; del buf214  # reuse
        buf279 = as_strided(buf273, (128, 128, 3072), (393216, 3072, 1)); del buf273  # reuse
        buf280 = as_strided(buf279, (16384, 3072), (3072, 1)); del buf279  # reuse
        buf283 = buf217; del buf217  # reuse
        triton_fused_sum_12_view_123_7.run(buf280, buf283, 24576, 2048, grid=grid(24576), stream=stream0)
        buf339 = buf280; del buf280  # reuse
        buf345 = as_strided(buf339, (128, 128, 3072), (393216, 3072, 1)); del buf339  # reuse
        buf346 = as_strided(buf345, (16384, 3072), (3072, 1)); del buf345  # reuse
        buf349 = buf283; del buf283  # reuse
        triton_fused_sum_12_view_123_7.run(buf346, buf349, 24576, 2048, grid=grid(24576), stream=stream0)
        del buf346
        return (buf349, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from intel_extension_for_pytorch._inductor.xpu.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    primals_102 = rand_strided((128, 128), (128, 1), device='xpu:0', dtype=torch.int64)
    embedding = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    slice_2 = rand_strided((1, 128), (512, 1), device='xpu:0', dtype=torch.int64)
    embedding_1 = rand_strided((1, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    getitem_1 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    rsqrt = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    philox_seed_like = rand_strided((), (), device='xpu:0', dtype=torch.int64)
    view = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    view_10 = rand_strided((128, 1, 1, 128), (128, 128, 128, 1), device='xpu:0', dtype=torch.bool)
    convert_element_type_12 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_14 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_6 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_16 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_4 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_18 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_13 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_20 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_35 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_34 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_17 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_36 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_10 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_38 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_24 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_40 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_58 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_54 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_28 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_56 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_16 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_58 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_35 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_60 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_81 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_74 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_39 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_76 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_22 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_78 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_46 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_80 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_104 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_94 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_50 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_96 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_28 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_98 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_57 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_100 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_127 = rand_strided((128, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_114 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_61 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_116 = rand_strided((16384, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_34 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_118 = rand_strided((16384, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_68 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    div_12 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_66 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_70 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_13 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_74 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_79 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_80 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_81 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_82 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_85 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_90 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_95 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_15 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_103 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_16 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_107 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_112 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_113 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_114 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_115 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_118 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_123 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_128 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_18 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_136 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_19 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_145 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_146 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_147 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_148 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_151 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_156 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_161 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_21 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_165 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_169 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_22 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_173 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_178 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_179 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_180 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_181 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_184 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_189 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_194 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_24 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_198 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_202 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_25 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_206 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_211 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_212 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_213 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_214 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_217 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_222 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_227 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_27 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_235 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_28 = rand_strided((128, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_244 = rand_strided((1536, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_245 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_246 = rand_strided((1536, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_247 = rand_strided((1536, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_250 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_255 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_260 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    tangents_1 = rand_strided((128, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_102, embedding, slice_2, embedding_1, getitem_1, rsqrt, philox_seed_like, view, view_10, convert_element_type_12, view_14, mul_6, view_16, addmm_4, view_18, mul_13, view_20, convert_element_type_35, view_34, mul_17, view_36, addmm_10, view_38, mul_24, view_40, convert_element_type_58, view_54, mul_28, view_56, addmm_16, view_58, mul_35, view_60, convert_element_type_81, view_74, mul_39, view_76, addmm_22, view_78, mul_46, view_80, convert_element_type_104, view_94, mul_50, view_96, addmm_28, view_98, mul_57, view_100, convert_element_type_127, view_114, mul_61, view_116, addmm_34, view_118, mul_68, div_12, permute_66, permute_70, div_13, permute_74, permute_79, permute_80, permute_81, permute_82, permute_85, permute_90, permute_95, div_15, permute_99, permute_103, div_16, permute_107, permute_112, permute_113, permute_114, permute_115, permute_118, permute_123, permute_128, div_18, permute_132, permute_136, div_19, permute_140, permute_145, permute_146, permute_147, permute_148, permute_151, permute_156, permute_161, div_21, permute_165, permute_169, div_22, permute_173, permute_178, permute_179, permute_180, permute_181, permute_184, permute_189, permute_194, div_24, permute_198, permute_202, div_25, permute_206, permute_211, permute_212, permute_213, permute_214, permute_217, permute_222, permute_227, div_27, permute_231, permute_235, div_28, permute_239, permute_244, permute_245, permute_246, permute_247, permute_250, permute_255, permute_260, tangents_1]))
