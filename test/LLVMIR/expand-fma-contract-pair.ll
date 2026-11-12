; RUN: triton-llvm-opt -expand-fma-contract-pair %s | FileCheck %s

; CHECK-LABEL: @fma_f32
define float @fma_f32(float %a, float %b, float %c) {
; CHECK-NOT: @llvm.fma
; The contract multiply must be the first operand of the add.
; CHECK:     [[MUL:%.*]] = fmul contract float %a, %b
; CHECK-NEXT:[[ADD:%.*]] = fadd contract float [[MUL]], %c
; CHECK-NEXT:ret float [[ADD]]
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; A separately rounded addend stays a plain multiply (no contract flag).
; CHECK-LABEL: @fma_f32_mulf_addend
define float @fma_f32_mulf_addend(float %a, float %b, float %x) {
; CHECK:     [[ADDEND:%.*]] = fmul float %x, 1.000000e-01
; CHECK:     [[MUL:%.*]] = fmul contract float %b, {{.*}}
; CHECK-NEXT:[[ADD:%.*]] = fadd contract float [[MUL]], [[ADDEND]]
  %addend = fmul float %x, 1.000000e-01
  %r = call float @llvm.fma.f32(float %b, float 0x3FECCCCCC0000000, float %addend)
  ret float %r
}

; Both multiplicands constant: folding the product would round it to f32 and
; change the result of an exact fma, so llvm.fma is kept.
; CHECK-LABEL: @fma_f32_const_multiplicands
define float @fma_f32_const_multiplicands(float %c) {
; CHECK: call float @llvm.fma.f32(float f0x3F800001, float f0x3F800001, float %c)
; CHECK-NOT: fmul contract
  %r = call float @llvm.fma.f32(float f0x3F800001, float f0x3F800001, float %c)
  ret float %r
}

; One constant multiplicand is still rewritten.
; CHECK-LABEL: @fma_f32_one_const_multiplicand
define float @fma_f32_one_const_multiplicand(float %a, float %c) {
; CHECK:     [[MUL:%.*]] = fmul contract float %a, {{.*}}
; CHECK-NEXT:fadd contract float [[MUL]], %c
  %r = call float @llvm.fma.f32(float %a, float f0x3F800001, float %c)
  ret float %r
}

; Only scalar f32 is rewritten.
; CHECK-LABEL: @fma_f64
define double @fma_f64(double %a, double %b, double %c) {
; CHECK: call double @llvm.fma.f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

; CHECK-LABEL: @fma_v4f32
define <4 x float> @fma_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; CHECK: call <4 x float> @llvm.fma.v4f32
  %r = call <4 x float> @llvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %r
}

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
