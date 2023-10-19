
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


triton_fused__unsafe_view_20__unsafe_view_24__unsafe_view_28__unsafe_view_32__unsafe_view_36__unsafe_view_40_29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_fused__unsafe_view_20__unsafe_view_24__unsafe_view_28__unsafe_view_32__unsafe_view_36__unsafe_view_40_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp1 = r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 + tmp14
        tmp16 = tmp5 * tmp15
        tmp17 = 1.1111111111111112
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask & xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    x2 = xindex % 128
    tmp28 = tl.load(in_ptr7 + (x0), xmask)
    tmp30 = tl.load(in_ptr8 + (x0), xmask)
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp25 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp26 = tl.load(in_ptr6 + (r1 + (768*x2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp24 = tmp22 * tmp23
        tmp27 = tmp25 + tmp26
        tmp29 = tmp27 - tmp28
        tmp31 = tmp29 * tmp30
        tmp32 = tmp24 * tmp31
        _tmp33 = tl.where(rmask & xmask, _tmp33 + tmp32, _tmp33)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp49 = tl.load(in_ptr9 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp41 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp42 = tl.load(in_ptr6 + (r1 + (768*x2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp34 = 768.0
        tmp35 = tmp30 / tmp34
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 * tmp34
        tmp40 = tmp39 - tmp21
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43 - tmp28
        tmp45 = tmp44 * tmp30
        tmp46 = tmp45 * tmp33
        tmp47 = tmp40 - tmp46
        tmp48 = tmp35 * tmp47
        tmp50 = 0
        tmp51 = tmp49 == tmp50
        tmp52 = 0.0
        tmp53 = tl.where(tmp51, tmp52, tmp48)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
        tl.atomic_add(out_ptr3 + (r1 + (768*tmp49) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp53, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_102, embedding, slice_2, embedding_1, getitem_1, rsqrt, philox_seed_like, view, view_10, convert_element_type_12, view_14, mul_6, view_16, addmm_4, view_18, mul_13, view_20, convert_element_type_35, view_34, mul_17, view_36, addmm_10, view_38, mul_24, view_40, convert_element_type_58, view_54, mul_28, view_56, addmm_16, view_58, mul_35, view_60, convert_element_type_81, view_74, mul_39, view_76, addmm_22, view_78, mul_46, view_80, convert_element_type_104, view_94, mul_50, view_96, addmm_28, view_98, mul_57, view_100, convert_element_type_127, view_114, mul_61, view_116, addmm_34, view_118, mul_68, div_12, permute_66, permute_70, div_13, permute_74, permute_79, permute_80, permute_81, permute_82, permute_85, permute_90, permute_95, div_15, permute_99, permute_103, div_16, permute_107, permute_112, permute_113, permute_114, permute_115, permute_118, permute_123, permute_128, div_18, permute_132, permute_136, div_19, permute_140, permute_145, permute_146, permute_147, permute_148, permute_151, permute_156, permute_161, div_21, permute_165, permute_169, div_22, permute_173, permute_178, permute_179, permute_180, permute_181, permute_184, permute_189, permute_194, div_24, permute_198, permute_202, div_25, permute_206, permute_211, permute_212, permute_213, permute_214, permute_217, permute_222, permute_227, div_27, permute_231, permute_235, div_28, permute_239, permute_244, permute_245, permute_246, permute_247, permute_250, permute_255, permute_260, tangents_1 = args
    args.clear()
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0) # no-op to ensure context
        stream0 = get_xpu_stream(0)

        buf25 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu', dtype=torch.float32)
        buf30 = rand_strided((32768, 768), (768, 1), device='xpu', dtype=torch.float16)
        buf37 = as_strided(buf30, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf30  # reuse
        buf42 = as_strided(buf37, (3072, 64, 128), (8192, 128, 1)); del buf37  # reuse
        buf43 = rand_strided((3072, 128, 64), (8192, 64, 1), device='xpu', dtype=torch.float16)
        buf44 = rand_strided((32768, 768), (768, 1), device='xpu', dtype=torch.float16)
        buf51 = buf44; del buf44  # reuse
        buf52 = as_strided(buf42, (32768, 768), (768, 1)); del buf42  # reuse
        buf58 = buf51; del buf51  # reuse
        buf59 = as_strided(buf43, (32768, 768), (768, 1)); del buf43  # reuse
        buf68 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu', dtype=torch.float32)
        buf73 = as_strided(buf58, (256, 128, 768), (98304, 768, 1)); del buf58  # reuse
        buf74 = as_strided(buf73, (32768, 768), (768, 1)); del buf73  # reuse
        buf83 = buf74; del buf74  # reuse
        buf91 = buf25; del buf25  # reuse
        buf96 = buf59; del buf59  # reuse
        buf97 = buf83; del buf83  # reuse
        buf103 = as_strided(buf96, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf96  # reuse
        buf104 = as_strided(buf97, (3072, 128, 64), (8192, 64, 1)); del buf97  # reuse
        buf108 = as_strided(buf103, (3072, 64, 128), (8192, 128, 1)); del buf103  # reuse
        buf109 = as_strided(buf52, (3072, 128, 64), (8192, 64, 1)); del buf52  # reuse
        buf111 = as_strided(buf104, (32768, 768), (768, 1)); del buf104  # reuse
        buf118 = as_strided(buf108, (32768, 768), (768, 1)); del buf108  # reuse
        buf125 = as_strided(buf109, (32768, 768), (768, 1)); del buf109  # reuse
        buf131 = buf68; del buf68  # reuse
        buf157 = buf91; del buf91  # reuse
        buf162 = buf125; del buf125  # reuse
        buf169 = as_strided(buf162, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf162  # reuse
        buf174 = as_strided(buf169, (3072, 64, 128), (8192, 128, 1)); del buf169  # reuse
        buf175 = as_strided(buf118, (3072, 128, 64), (8192, 64, 1)); del buf118  # reuse
        buf176 = buf111; del buf111  # reuse
        buf183 = buf176; del buf176  # reuse
        buf184 = as_strided(buf174, (32768, 768), (768, 1)); del buf174  # reuse
        buf190 = buf183; del buf183  # reuse
        buf191 = as_strided(buf175, (32768, 768), (768, 1)); del buf175  # reuse
        buf200 = buf131; del buf131  # reuse
        buf205 = as_strided(buf190, (256, 128, 768), (98304, 768, 1)); del buf190  # reuse
        buf206 = as_strided(buf205, (32768, 768), (768, 1)); del buf205  # reuse
        buf215 = buf206; del buf206  # reuse
        buf223 = buf157; del buf157  # reuse
        buf228 = buf191; del buf191  # reuse
        buf229 = buf215; del buf215  # reuse
        buf235 = as_strided(buf228, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf228  # reuse
        buf236 = as_strided(buf229, (3072, 128, 64), (8192, 64, 1)); del buf229  # reuse
        buf240 = as_strided(buf235, (3072, 64, 128), (8192, 128, 1)); del buf235  # reuse
        buf241 = as_strided(buf184, (3072, 128, 64), (8192, 64, 1)); del buf184  # reuse
        buf243 = as_strided(buf236, (32768, 768), (768, 1)); del buf236  # reuse
        buf250 = as_strided(buf240, (32768, 768), (768, 1)); del buf240  # reuse
        buf257 = as_strided(buf241, (32768, 768), (768, 1)); del buf241  # reuse
        buf263 = buf200; del buf200  # reuse
        buf289 = buf223; del buf223  # reuse
        buf294 = buf257; del buf257  # reuse
        buf301 = as_strided(buf294, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf294  # reuse
        buf306 = as_strided(buf301, (3072, 64, 128), (8192, 128, 1)); del buf301  # reuse
        buf307 = as_strided(buf250, (3072, 128, 64), (8192, 64, 1)); del buf250  # reuse
        buf308 = buf243; del buf243  # reuse
        buf315 = buf308; del buf308  # reuse
        buf316 = as_strided(buf306, (32768, 768), (768, 1)); del buf306  # reuse
        buf322 = buf315; del buf315  # reuse
        buf323 = as_strided(buf307, (32768, 768), (768, 1)); del buf307  # reuse
        buf332 = buf263; del buf263  # reuse
        buf337 = as_strided(buf322, (256, 128, 768), (98304, 768, 1)); del buf322  # reuse
        buf338 = as_strided(buf337, (32768, 768), (768, 1)); del buf337  # reuse
        buf347 = buf338; del buf338  # reuse
        buf355 = buf289; del buf289  # reuse
        buf360 = buf323; del buf323  # reuse
        buf361 = buf347; del buf347  # reuse
        buf367 = as_strided(buf360, (256, 12, 128, 64), (98304, 8192, 64, 1)); del buf360  # reuse
        buf368 = as_strided(buf361, (3072, 128, 64), (8192, 64, 1)); del buf361  # reuse
        buf372 = as_strided(buf367, (3072, 64, 128), (8192, 128, 1)); del buf367  # reuse
        buf373 = as_strided(buf316, (3072, 128, 64), (8192, 64, 1)); del buf316  # reuse
        buf375 = as_strided(buf368, (32768, 768), (768, 1)); del buf368  # reuse
        buf382 = as_strided(buf372, (32768, 768), (768, 1)); del buf372  # reuse
        buf389 = as_strided(buf373, (32768, 768), (768, 1)); del buf373  # reuse
        buf406 = rand_strided((30522, 768), (768, 1), device='xpu', dtype=torch.float32)
        buf395 = buf355; del buf355  # reuse
        buf398 = buf332; del buf332  # reuse
        triton_fused__unsafe_view_20__unsafe_view_24__unsafe_view_28__unsafe_view_32__unsafe_view_36__unsafe_view_40_29.run(buf395, philox_seed_like, buf375, buf382, buf389, primals_3, embedding, embedding_1, getitem_1, rsqrt, primals_102, buf398, buf406, 32768, 768, grid=grid(32768), stream=stream0)
        del buf375
        del buf382
        del buf389
        del philox_seed_like
        del primals_102
        del primals_3
        del embedding
        del embedding_1
        del getitem_1
        del rsqrt
        return (buf398, buf406)


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
    primals_102 = rand_strided((256, 128), (128, 1), device='xpu:0', dtype=torch.int64)
    embedding = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    slice_2 = rand_strided((1, 128), (512, 1), device='xpu:0', dtype=torch.int64)
    embedding_1 = rand_strided((1, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    getitem_1 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    rsqrt = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    philox_seed_like = rand_strided((), (), device='xpu:0', dtype=torch.int64)
    view = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    view_10 = rand_strided((256, 1, 1, 128), (128, 128, 128, 1), device='xpu:0', dtype=torch.bool)
    convert_element_type_12 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_14 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_6 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_16 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_4 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_18 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_13 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_20 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_35 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_34 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_17 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_36 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_10 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_38 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_24 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_40 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_58 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_54 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_28 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_56 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_16 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_58 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_35 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_60 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_81 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_74 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_39 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_76 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_22 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_78 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_46 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_80 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_104 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_94 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_50 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_96 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_28 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_98 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_57 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_100 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    convert_element_type_127 = rand_strided((256, 12, 128, 128), (196608, 16384, 128, 1), device='xpu:0', dtype=torch.float16)
    view_114 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    mul_61 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    view_116 = rand_strided((32768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    addmm_34 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    view_118 = rand_strided((32768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    mul_68 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    div_12 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_66 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_70 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_13 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_74 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_79 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_80 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_81 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_82 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_85 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_90 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_95 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_15 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_103 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_16 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_107 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_112 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_113 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_114 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_115 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_118 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_123 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_128 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_18 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_136 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_19 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_145 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_146 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_147 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_148 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_151 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_156 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_161 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_21 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_165 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_169 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_22 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_173 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_178 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_179 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_180 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_181 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_184 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_189 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_194 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_24 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_198 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_202 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_25 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_206 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_211 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_212 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_213 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_214 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_217 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_222 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_227 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_27 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 3072), (3072, 1), device='xpu:0', dtype=torch.float16)
    permute_235 = rand_strided((3072, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    div_28 = rand_strided((256, 128, 1), (128, 1, 1), device='xpu:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_244 = rand_strided((3072, 128, 128), (16384, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_245 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_246 = rand_strided((3072, 64, 128), (8192, 1, 64), device='xpu:0', dtype=torch.float16)
    permute_247 = rand_strided((3072, 128, 64), (8192, 1, 128), device='xpu:0', dtype=torch.float16)
    permute_250 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_255 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    permute_260 = rand_strided((768, 768), (768, 1), device='xpu:0', dtype=torch.float16)
    tangents_1 = rand_strided((256, 128, 768), (98304, 768, 1), device='xpu:0', dtype=torch.float32)
    print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_102, embedding, slice_2, embedding_1, getitem_1, rsqrt, philox_seed_like, view, view_10, convert_element_type_12, view_14, mul_6, view_16, addmm_4, view_18, mul_13, view_20, convert_element_type_35, view_34, mul_17, view_36, addmm_10, view_38, mul_24, view_40, convert_element_type_58, view_54, mul_28, view_56, addmm_16, view_58, mul_35, view_60, convert_element_type_81, view_74, mul_39, view_76, addmm_22, view_78, mul_46, view_80, convert_element_type_104, view_94, mul_50, view_96, addmm_28, view_98, mul_57, view_100, convert_element_type_127, view_114, mul_61, view_116, addmm_34, view_118, mul_68, div_12, permute_66, permute_70, div_13, permute_74, permute_79, permute_80, permute_81, permute_82, permute_85, permute_90, permute_95, div_15, permute_99, permute_103, div_16, permute_107, permute_112, permute_113, permute_114, permute_115, permute_118, permute_123, permute_128, div_18, permute_132, permute_136, div_19, permute_140, permute_145, permute_146, permute_147, permute_148, permute_151, permute_156, permute_161, div_21, permute_165, permute_169, div_22, permute_173, permute_178, permute_179, permute_180, permute_181, permute_184, permute_189, permute_194, div_24, permute_198, permute_202, div_25, permute_206, permute_211, permute_212, permute_213, permute_214, permute_217, permute_222, permute_227, div_27, permute_231, permute_235, div_28, permute_239, permute_244, permute_245, permute_246, permute_247, permute_250, permute_255, permute_260, tangents_1]))

