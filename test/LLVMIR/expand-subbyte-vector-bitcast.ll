; RUN: triton-llvm-opt -expand-subbyte-vector-bitcast %s | FileCheck %s

; A `bitcast <K x i1> to <M x iN>` (N<8) read back via extractelement and tested
; with icmp eq/ne <const> is rewritten onto the source i1 lanes, so no sub-byte
; integer type (scalar iN or <M x iN> vector) reaches the SPIR-V translator

; "any of the first two lanes set": icmp ne <lane0>, 0 lowers to
; not((not bit0) and (not bit1)), i.e. bit0 or bit1, over the source i1 lanes.
; CHECK-LABEL: @any_of_first_two
define i1 @any_of_first_two(<4 x float> %v) {
; CHECK-NOT:   bitcast <4 x i1> {{.*}} to <2 x i2>
; CHECK-NOT:   extractelement <2 x i2>
; CHECK-NOT:    i2
; CHECK:       [[CMP:%.*]] = fcmp ule <4 x float> %v,
; CHECK:       [[B0:%.*]] = extractelement <4 x i1> [[CMP]], i64 0
; CHECK:       [[N0:%.*]] = xor i1 [[B0]], true
; CHECK:       [[B1:%.*]] = extractelement <4 x i1> [[CMP]], i64 1
; CHECK:       [[N1:%.*]] = xor i1 [[B1]], true
; CHECK:       [[AND:%.*]] = and i1 [[N0]], [[N1]]
; CHECK:       [[NE:%.*]] = xor i1 [[AND]], true
; CHECK:       ret i1 [[NE]]
  %cmp = fcmp ule <4 x float> %v, <float 1.0, float 1.0, float poison, float poison>
  %bc = bitcast <4 x i1> %cmp to <2 x i2>
  %e = extractelement <2 x i2> %bc, i64 0
  %ne = icmp ne i2 %e, 0
  ret i1 %ne
}

; icmp eq <lane1>, 3 (both bits set): (bit2 and bit3).
; CHECK-LABEL: @lane_one_eq_three
define i1 @lane_one_eq_three(<4 x i1> %bits) {
; CHECK-NOT:    i2
; CHECK:       [[B2:%.*]] = extractelement <4 x i1> %bits, i64 2
; CHECK:       [[B3:%.*]] = extractelement <4 x i1> %bits, i64 3
; CHECK:       [[AND:%.*]] = and i1 [[B2]], [[B3]]
; CHECK:       ret i1 [[AND]]
  %bc = bitcast <4 x i1> %bits to <2 x i2>
  %e = extractelement <2 x i2> %bc, i64 1
  %eq = icmp eq i2 %e, 3
  ret i1 %eq
}

; A wider (>=8-bit) integer vector bitcast is left untouched.
; CHECK-LABEL: @vec_i8_untouched
define i8 @vec_i8_untouched(<4 x i16> %v) {
; CHECK:       [[BC:%.*]] = bitcast <4 x i16> %v to <8 x i8>
; CHECK:       extractelement <8 x i8> [[BC]]
  %bc = bitcast <4 x i16> %v to <8 x i8>
  %e = extractelement <8 x i8> %bc, i64 0
  ret i8 %e
}
