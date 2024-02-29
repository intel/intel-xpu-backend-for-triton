
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
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*fp32', 14: '*fp32', 15: '*bf16', 16: '*bf16', 17: '*bf16', 18: '*bf16', 19: '*bf16', 20: '*fp32', 21: '*fp32', 22: '*bf16', 23: '*bf16', 24: '*bf16', 25: '*bf16', 26: '*bf16', 27: '*fp32', 28: '*fp32', 29: '*bf16', 30: '*bf16', 31: '*bf16', 32: '*bf16', 33: '*bf16', 34: '*fp32', 35: '*fp32', 36: '*bf16', 37: '*bf16', 38: '*bf16', 39: '*bf16', 40: '*bf16', 41: '*fp32', 42: '*fp32', 43: '*bf16', 44: '*bf16', 45: '*bf16', 46: '*bf16', 47: '*bf16', 48: '*fp32', 49: '*fp32', 50: '*bf16', 51: '*bf16', 52: '*bf16', 53: '*bf16', 54: '*bf16', 55: '*fp32', 56: '*fp32', 57: '*bf16', 58: '*bf16', 59: '*bf16', 60: '*bf16', 61: '*bf16', 62: '*fp32', 63: '*fp32', 64: '*bf16', 65: '*bf16', 66: '*bf16', 67: '*bf16', 68: '*bf16', 69: '*fp32', 70: '*fp32', 71: '*bf16', 72: '*bf16', 73: '*bf16', 74: '*bf16', 75: '*bf16', 76: '*fp32', 77: '*fp32', 78: '*bf16', 79: '*bf16', 80: '*bf16', 81: '*bf16', 82: '*bf16', 83: '*fp32', 84: '*fp32', 85: 'i32', 86: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_sum_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_sum_19(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp45 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp62 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp65 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp69 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp86 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp89 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp93 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp110 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp113 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp117 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp134 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp137 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp141 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp158 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp161 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp165 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp182 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp185 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp189 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp206 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp209 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp213 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp230 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp233 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp237 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp254 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp257 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp261 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp278 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp281 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp19 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp25 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp27 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp32 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp34 = tl.load(in_ptr11 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp43 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp47 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp49 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp51 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp54 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr17 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp58 = tl.load(in_ptr18 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp67 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp71 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp73 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp75 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp78 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp80 = tl.load(in_ptr24 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp82 = tl.load(in_ptr25 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp91 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp95 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp97 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp99 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp102 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp104 = tl.load(in_ptr31 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp106 = tl.load(in_ptr32 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp115 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp119 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp121 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp123 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp126 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp128 = tl.load(in_ptr38 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp130 = tl.load(in_ptr39 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp139 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp145 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp147 = tl.load(in_ptr43 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp150 = tl.load(in_ptr44 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp152 = tl.load(in_ptr45 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp154 = tl.load(in_ptr46 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp163 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp167 = tl.load(in_ptr48 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp169 = tl.load(in_ptr49 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp171 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp174 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp176 = tl.load(in_ptr52 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp178 = tl.load(in_ptr53 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp187 = tl.load(in_ptr54 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp191 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp193 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp195 = tl.load(in_ptr57 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp198 = tl.load(in_ptr58 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp200 = tl.load(in_ptr59 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp202 = tl.load(in_ptr60 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp211 = tl.load(in_ptr61 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp215 = tl.load(in_ptr62 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp217 = tl.load(in_ptr63 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp219 = tl.load(in_ptr64 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp222 = tl.load(in_ptr65 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp224 = tl.load(in_ptr66 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp226 = tl.load(in_ptr67 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp235 = tl.load(in_ptr68 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp239 = tl.load(in_ptr69 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp241 = tl.load(in_ptr70 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp243 = tl.load(in_ptr71 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp246 = tl.load(in_ptr72 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp248 = tl.load(in_ptr73 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp250 = tl.load(in_ptr74 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp259 = tl.load(in_ptr75 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp263 = tl.load(in_ptr76 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp265 = tl.load(in_ptr77 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp267 = tl.load(in_ptr78 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp270 = tl.load(in_ptr79 + (x0 + (4096*r1)), rmask, other=0).to(tl.float32)
        tmp272 = tl.load(in_ptr80 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp274 = tl.load(in_ptr81 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
        tmp24 = tmp19 + tmp23
        tmp26 = tmp24 + tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 - tmp32
        tmp35 = tmp33 * tmp34
        tmp36 = tmp29 * tmp35
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp40 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
        tmp46 = _tmp45 + tmp44
        _tmp45 = tl.where(rmask, tmp46, _tmp45)
        tmp48 = tmp43 + tmp47
        tmp50 = tmp48 + tmp49
        tmp52 = tmp50 + tmp51
        tmp53 = tmp52.to(tl.float32)
        tmp55 = tmp54.to(tl.float32)
        tmp57 = tmp55 - tmp56
        tmp59 = tmp57 * tmp58
        tmp60 = tmp53 * tmp59
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
        tmp63 = _tmp62 + tmp61
        _tmp62 = tl.where(rmask, tmp63, _tmp62)
        tmp64 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(rmask, tmp66, _tmp65)
        tmp68 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
        tmp70 = _tmp69 + tmp68
        _tmp69 = tl.where(rmask, tmp70, _tmp69)
        tmp72 = tmp67 + tmp71
        tmp74 = tmp72 + tmp73
        tmp76 = tmp74 + tmp75
        tmp77 = tmp76.to(tl.float32)
        tmp79 = tmp78.to(tl.float32)
        tmp81 = tmp79 - tmp80
        tmp83 = tmp81 * tmp82
        tmp84 = tmp77 * tmp83
        tmp85 = tl.broadcast_to(tmp84, [XBLOCK, RBLOCK])
        tmp87 = _tmp86 + tmp85
        _tmp86 = tl.where(rmask, tmp87, _tmp86)
        tmp88 = tl.broadcast_to(tmp77, [XBLOCK, RBLOCK])
        tmp90 = _tmp89 + tmp88
        _tmp89 = tl.where(rmask, tmp90, _tmp89)
        tmp92 = tl.broadcast_to(tmp91, [XBLOCK, RBLOCK])
        tmp94 = _tmp93 + tmp92
        _tmp93 = tl.where(rmask, tmp94, _tmp93)
        tmp96 = tmp91 + tmp95
        tmp98 = tmp96 + tmp97
        tmp100 = tmp98 + tmp99
        tmp101 = tmp100.to(tl.float32)
        tmp103 = tmp102.to(tl.float32)
        tmp105 = tmp103 - tmp104
        tmp107 = tmp105 * tmp106
        tmp108 = tmp101 * tmp107
        tmp109 = tl.broadcast_to(tmp108, [XBLOCK, RBLOCK])
        tmp111 = _tmp110 + tmp109
        _tmp110 = tl.where(rmask, tmp111, _tmp110)
        tmp112 = tl.broadcast_to(tmp101, [XBLOCK, RBLOCK])
        tmp114 = _tmp113 + tmp112
        _tmp113 = tl.where(rmask, tmp114, _tmp113)
        tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
        tmp118 = _tmp117 + tmp116
        _tmp117 = tl.where(rmask, tmp118, _tmp117)
        tmp120 = tmp115 + tmp119
        tmp122 = tmp120 + tmp121
        tmp124 = tmp122 + tmp123
        tmp125 = tmp124.to(tl.float32)
        tmp127 = tmp126.to(tl.float32)
        tmp129 = tmp127 - tmp128
        tmp131 = tmp129 * tmp130
        tmp132 = tmp125 * tmp131
        tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
        tmp135 = _tmp134 + tmp133
        _tmp134 = tl.where(rmask, tmp135, _tmp134)
        tmp136 = tl.broadcast_to(tmp125, [XBLOCK, RBLOCK])
        tmp138 = _tmp137 + tmp136
        _tmp137 = tl.where(rmask, tmp138, _tmp137)
        tmp140 = tl.broadcast_to(tmp139, [XBLOCK, RBLOCK])
        tmp142 = _tmp141 + tmp140
        _tmp141 = tl.where(rmask, tmp142, _tmp141)
        tmp144 = tmp139 + tmp143
        tmp146 = tmp144 + tmp145
        tmp148 = tmp146 + tmp147
        tmp149 = tmp148.to(tl.float32)
        tmp151 = tmp150.to(tl.float32)
        tmp153 = tmp151 - tmp152
        tmp155 = tmp153 * tmp154
        tmp156 = tmp149 * tmp155
        tmp157 = tl.broadcast_to(tmp156, [XBLOCK, RBLOCK])
        tmp159 = _tmp158 + tmp157
        _tmp158 = tl.where(rmask, tmp159, _tmp158)
        tmp160 = tl.broadcast_to(tmp149, [XBLOCK, RBLOCK])
        tmp162 = _tmp161 + tmp160
        _tmp161 = tl.where(rmask, tmp162, _tmp161)
        tmp164 = tl.broadcast_to(tmp163, [XBLOCK, RBLOCK])
        tmp166 = _tmp165 + tmp164
        _tmp165 = tl.where(rmask, tmp166, _tmp165)
        tmp168 = tmp163 + tmp167
        tmp170 = tmp168 + tmp169
        tmp172 = tmp170 + tmp171
        tmp173 = tmp172.to(tl.float32)
        tmp175 = tmp174.to(tl.float32)
        tmp177 = tmp175 - tmp176
        tmp179 = tmp177 * tmp178
        tmp180 = tmp173 * tmp179
        tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
        tmp183 = _tmp182 + tmp181
        _tmp182 = tl.where(rmask, tmp183, _tmp182)
        tmp184 = tl.broadcast_to(tmp173, [XBLOCK, RBLOCK])
        tmp186 = _tmp185 + tmp184
        _tmp185 = tl.where(rmask, tmp186, _tmp185)
        tmp188 = tl.broadcast_to(tmp187, [XBLOCK, RBLOCK])
        tmp190 = _tmp189 + tmp188
        _tmp189 = tl.where(rmask, tmp190, _tmp189)
        tmp192 = tmp187 + tmp191
        tmp194 = tmp192 + tmp193
        tmp196 = tmp194 + tmp195
        tmp197 = tmp196.to(tl.float32)
        tmp199 = tmp198.to(tl.float32)
        tmp201 = tmp199 - tmp200
        tmp203 = tmp201 * tmp202
        tmp204 = tmp197 * tmp203
        tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
        tmp207 = _tmp206 + tmp205
        _tmp206 = tl.where(rmask, tmp207, _tmp206)
        tmp208 = tl.broadcast_to(tmp197, [XBLOCK, RBLOCK])
        tmp210 = _tmp209 + tmp208
        _tmp209 = tl.where(rmask, tmp210, _tmp209)
        tmp212 = tl.broadcast_to(tmp211, [XBLOCK, RBLOCK])
        tmp214 = _tmp213 + tmp212
        _tmp213 = tl.where(rmask, tmp214, _tmp213)
        tmp216 = tmp211 + tmp215
        tmp218 = tmp216 + tmp217
        tmp220 = tmp218 + tmp219
        tmp221 = tmp220.to(tl.float32)
        tmp223 = tmp222.to(tl.float32)
        tmp225 = tmp223 - tmp224
        tmp227 = tmp225 * tmp226
        tmp228 = tmp221 * tmp227
        tmp229 = tl.broadcast_to(tmp228, [XBLOCK, RBLOCK])
        tmp231 = _tmp230 + tmp229
        _tmp230 = tl.where(rmask, tmp231, _tmp230)
        tmp232 = tl.broadcast_to(tmp221, [XBLOCK, RBLOCK])
        tmp234 = _tmp233 + tmp232
        _tmp233 = tl.where(rmask, tmp234, _tmp233)
        tmp236 = tl.broadcast_to(tmp235, [XBLOCK, RBLOCK])
        tmp238 = _tmp237 + tmp236
        _tmp237 = tl.where(rmask, tmp238, _tmp237)
        tmp240 = tmp235 + tmp239
        tmp242 = tmp240 + tmp241
        tmp244 = tmp242 + tmp243
        tmp245 = tmp244.to(tl.float32)
        tmp247 = tmp246.to(tl.float32)
        tmp249 = tmp247 - tmp248
        tmp251 = tmp249 * tmp250
        tmp252 = tmp245 * tmp251
        tmp253 = tl.broadcast_to(tmp252, [XBLOCK, RBLOCK])
        tmp255 = _tmp254 + tmp253
        _tmp254 = tl.where(rmask, tmp255, _tmp254)
        tmp256 = tl.broadcast_to(tmp245, [XBLOCK, RBLOCK])
        tmp258 = _tmp257 + tmp256
        _tmp257 = tl.where(rmask, tmp258, _tmp257)
        tmp260 = tl.broadcast_to(tmp259, [XBLOCK, RBLOCK])
        tmp262 = _tmp261 + tmp260
        _tmp261 = tl.where(rmask, tmp262, _tmp261)
        tmp264 = tmp259 + tmp263
        tmp266 = tmp264 + tmp265
        tmp268 = tmp266 + tmp267
        tmp269 = tmp268.to(tl.float32)
        tmp271 = tmp270.to(tl.float32)
        tmp273 = tmp271 - tmp272
        tmp275 = tmp273 * tmp274
        tmp276 = tmp269 * tmp275
        tmp277 = tl.broadcast_to(tmp276, [XBLOCK, RBLOCK])
        tmp279 = _tmp278 + tmp277
        _tmp278 = tl.where(rmask, tmp279, _tmp278)
        tmp280 = tl.broadcast_to(tmp269, [XBLOCK, RBLOCK])
        tmp282 = _tmp281 + tmp280
        _tmp281 = tl.where(rmask, tmp282, _tmp281)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp45 = tl.sum(_tmp45, 1)[:, None]
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp69 = tl.sum(_tmp69, 1)[:, None]
    tmp86 = tl.sum(_tmp86, 1)[:, None]
    tmp89 = tl.sum(_tmp89, 1)[:, None]
    tmp93 = tl.sum(_tmp93, 1)[:, None]
    tmp110 = tl.sum(_tmp110, 1)[:, None]
    tmp113 = tl.sum(_tmp113, 1)[:, None]
    tmp117 = tl.sum(_tmp117, 1)[:, None]
    tmp134 = tl.sum(_tmp134, 1)[:, None]
    tmp137 = tl.sum(_tmp137, 1)[:, None]
    tmp141 = tl.sum(_tmp141, 1)[:, None]
    tmp158 = tl.sum(_tmp158, 1)[:, None]
    tmp161 = tl.sum(_tmp161, 1)[:, None]
    tmp165 = tl.sum(_tmp165, 1)[:, None]
    tmp182 = tl.sum(_tmp182, 1)[:, None]
    tmp185 = tl.sum(_tmp185, 1)[:, None]
    tmp189 = tl.sum(_tmp189, 1)[:, None]
    tmp206 = tl.sum(_tmp206, 1)[:, None]
    tmp209 = tl.sum(_tmp209, 1)[:, None]
    tmp213 = tl.sum(_tmp213, 1)[:, None]
    tmp230 = tl.sum(_tmp230, 1)[:, None]
    tmp233 = tl.sum(_tmp233, 1)[:, None]
    tmp237 = tl.sum(_tmp237, 1)[:, None]
    tmp254 = tl.sum(_tmp254, 1)[:, None]
    tmp257 = tl.sum(_tmp257, 1)[:, None]
    tmp261 = tl.sum(_tmp261, 1)[:, None]
    tmp278 = tl.sum(_tmp278, 1)[:, None]
    tmp281 = tl.sum(_tmp281, 1)[:, None]
    tmp283 = tmp14 + tmp38
    tmp284 = tmp283 + tmp62
    tmp285 = tmp284 + tmp86
    tmp286 = tmp285 + tmp110
    tmp287 = tmp286 + tmp134
    tmp288 = tmp287 + tmp182
    tmp289 = tmp288 + tmp230
    tmp290 = tmp289 + tmp278
    tmp291 = tmp290 + tmp158
    tmp292 = tmp291 + tmp206
    tmp293 = tmp292 + tmp254
    tmp294 = tmp17 + tmp41
    tmp295 = tmp294 + tmp65
    tmp296 = tmp295 + tmp89
    tmp297 = tmp296 + tmp113
    tmp298 = tmp297 + tmp137
    tmp299 = tmp298 + tmp185
    tmp300 = tmp299 + tmp233
    tmp301 = tmp300 + tmp281
    tmp302 = tmp301 + tmp161
    tmp303 = tmp302 + tmp209
    tmp304 = tmp303 + tmp257
    tmp305 = tmp21.to(tl.float32)
    tmp306 = tmp45.to(tl.float32)
    tmp307 = tmp305 + tmp306
    tmp308 = tmp69.to(tl.float32)
    tmp309 = tmp307 + tmp308
    tmp310 = tmp93.to(tl.float32)
    tmp311 = tmp309 + tmp310
    tmp312 = tmp117.to(tl.float32)
    tmp313 = tmp311 + tmp312
    tmp314 = tmp165.to(tl.float32)
    tmp315 = tmp313 + tmp314
    tmp316 = tmp213.to(tl.float32)
    tmp317 = tmp315 + tmp316
    tmp318 = tmp261.to(tl.float32)
    tmp319 = tmp317 + tmp318
    tmp320 = tmp141.to(tl.float32)
    tmp321 = tmp319 + tmp320
    tmp322 = tmp189.to(tl.float32)
    tmp323 = tmp321 + tmp322
    tmp324 = tmp237.to(tl.float32)
    tmp325 = tmp323 + tmp324
    tmp326 = tmp2.to(tl.float32)
    tmp327 = tmp325 + tmp326
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp293, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp304, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp327, None)


def get_args():
    arg_0 = rand_strided((4096,), (1,), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((4096,), (1,), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((4096,), (1,), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_11 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_12 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_13 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_14 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_15 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_16 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_17 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_18 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_19 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_20 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_21 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_22 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_23 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_24 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_25 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_26 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_27 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_28 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_29 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_30 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_31 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_32 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_33 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_34 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_35 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_36 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_37 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_38 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_39 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_40 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_41 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_42 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_43 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_44 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_45 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_46 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_47 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_48 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_49 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_50 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_51 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_52 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_53 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_54 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_55 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_56 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_57 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_58 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_59 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_60 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_61 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_62 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_63 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_64 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_65 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_66 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_67 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_68 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_69 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_70 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_71 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_72 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_73 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_74 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_75 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_76 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_77 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_78 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_79 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_80 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_81 = rand_strided((2048, 4096), (4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_82 = rand_strided((4, 512, 4096), (2097152, 4096, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_83 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_84 = rand_strided((4, 512, 1), (512, 1, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, arg_31, arg_32, arg_33, arg_34, arg_35, arg_36, arg_37, arg_38, arg_39, arg_40, arg_41, arg_42, arg_43, arg_44, arg_45, arg_46, arg_47, arg_48, arg_49, arg_50, arg_51, arg_52, arg_53, arg_54, arg_55, arg_56, arg_57, arg_58, arg_59, arg_60, arg_61, arg_62, arg_63, arg_64, arg_65, arg_66, arg_67, arg_68, arg_69, arg_70, arg_71, arg_72, arg_73, arg_74, arg_75, arg_76, arg_77, arg_78, arg_79, arg_80, arg_81, arg_82, arg_83, arg_84,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_sum_19.run(*args, 4096, 2048, grid=grid(4096), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_sum_19.benchmark_all_configs(*args, 4096, 2048, grid=grid(4096))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=3) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
