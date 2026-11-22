; RUN: triton-llvm-opt -expand-bf16-fpext-under-fabs %s | FileCheck %s

; CHECK-LABEL: @fabs_of_fpext
define float @fabs_of_fpext(bfloat %x) {
; CHECK-NOT: fpext
; CHECK:     [[BITS:%.*]] = bitcast bfloat %x to i16
; CHECK-NEXT:[[WIDE:%.*]] = zext i16 [[BITS]] to i32
; CHECK-NEXT:[[SHL:%.*]] = shl i32 [[WIDE]], 16
; CHECK-NEXT:[[F:%.*]] = bitcast i32 [[SHL]] to float
; CHECK-NEXT:[[ABS:%.*]] = call float @llvm.fabs.f32(float [[F]])
; CHECK-NEXT:ret float [[ABS]]
  %e = fpext bfloat %x to float
  %r = call float @llvm.fabs.f32(float %e)
  ret float %r
}

; CHECK-LABEL: @fpext_of_fabs
define <2 x float> @fpext_of_fabs(<2 x bfloat> %x) {
; CHECK-NOT: fpext
; CHECK-NOT: @llvm.fabs.v2bf16
; CHECK:     [[BITS:%.*]] = bitcast <2 x bfloat> %x to <2 x i16>
; CHECK-NEXT:[[WIDE:%.*]] = zext <2 x i16> [[BITS]] to <2 x i32>
; CHECK-NEXT:[[SHL:%.*]] = shl <2 x i32> [[WIDE]], splat (i32 16)
; CHECK-NEXT:[[F:%.*]] = bitcast <2 x i32> [[SHL]] to <2 x float>
; CHECK-NEXT:[[ABS:%.*]] = call <2 x float> @llvm.fabs.v2f32(<2 x float> [[F]])
; CHECK-NEXT:ret <2 x float> [[ABS]]
  %a = call <2 x bfloat> @llvm.fabs.v2bf16(<2 x bfloat> %x)
  %r = fpext <2 x bfloat> %a to <2 x float>
  ret <2 x float> %r
}

; CHECK-LABEL: @frem_divisor
define float @frem_divisor(float %x, bfloat %y) {
; CHECK-NOT: fpext
; CHECK:     [[F:%.*]] = bitcast i32 %{{.*}} to float
; CHECK-NEXT:[[R:%.*]] = frem float %x, [[F]]
  %e = fpext bfloat %y to float
  %r = frem float %x, %e
  ret float %r
}

; An fpext without an fabs, and fp16, are not touched.
; CHECK-LABEL: @untouched
define float @untouched(bfloat %x, half %y) {
; CHECK:     fpext bfloat %x to float
; CHECK:     call half @llvm.fabs.f16(half %y)
; CHECK-NEXT:fpext half
  %e = fpext bfloat %x to float
  %a = call half @llvm.fabs.f16(half %y)
  %h = fpext half %a to float
  %r = fadd float %e, %h
  ret float %r
}
