; RUN: triton-llvm-opt -expand-subbyte-bitreverse %s | FileCheck %s

; CHECK-LABEL: @bitrev_i4
define i4 @bitrev_i4(i4 %x) {
; CHECK-NOT: call i4 @llvm.bitreverse.i4
; CHECK:     [[ZEXT:%.*]] = zext i4 %x to i32
; CHECK:     [[REV:%.*]]  = call i32 @llvm.bitreverse.i32(i32 [[ZEXT]])
; CHECK:     [[SHR:%.*]]  = lshr i32 [[REV]], 28
; CHECK:     [[RES:%.*]]  = trunc i32 [[SHR]] to i4
; CHECK:     ret i4 [[RES]]
  %r = call i4 @llvm.bitreverse.i4(i4 %x)
  ret i4 %r
}

; CHECK-LABEL: @bitrev_i3
define i3 @bitrev_i3(i3 %x) {
; CHECK-NOT: call i3 @llvm.bitreverse.i3
; CHECK:     [[ZEXT:%.*]] = zext i3 %x to i32
; CHECK:     [[REV:%.*]]  = call i32 @llvm.bitreverse.i32(i32 [[ZEXT]])
; CHECK:     [[SHR:%.*]]  = lshr i32 [[REV]], 29
; CHECK:     [[RES:%.*]]  = trunc i32 [[SHR]] to i3
; CHECK:     ret i3 [[RES]]
  %r = call i3 @llvm.bitreverse.i3(i3 %x)
  ret i3 %r
}

; Ensure >=8-bit widths are not touched.
; CHECK-LABEL: @bitrev_i8
define i8 @bitrev_i8(i8 %x) {
; CHECK: call i8 @llvm.bitreverse.i8(i8 %x)
  %r = call i8 @llvm.bitreverse.i8(i8 %x)
  ret i8 %r
}

; CHECK-LABEL: @bitrev_i32
define i32 @bitrev_i32(i32 %x) {
; CHECK: call i32 @llvm.bitreverse.i32(i32 %x)
  %r = call i32 @llvm.bitreverse.i32(i32 %x)
  ret i32 %r
}

; Vector of sub-byte integers is widened element-wise via <K x i32>.
; CHECK-LABEL: @bitrev_v4i4
define <4 x i4> @bitrev_v4i4(<4 x i4> %x) {
; CHECK-NOT: call <4 x i4> @llvm.bitreverse.v4i4
; CHECK:     [[ZEXT:%.*]] = zext <4 x i4> %x to <4 x i32>
; CHECK:     [[REV:%.*]]  = call <4 x i32> @llvm.bitreverse.v4i32(<4 x i32> [[ZEXT]])
; CHECK:     [[SHR:%.*]]  = lshr <4 x i32> [[REV]], {{.*}}
; CHECK:     [[RES:%.*]]  = trunc <4 x i32> [[SHR]] to <4 x i4>
; CHECK:     ret <4 x i4> [[RES]]
  %r = call <4 x i4> @llvm.bitreverse.v4i4(<4 x i4> %x)
  ret <4 x i4> %r
}

declare i3  @llvm.bitreverse.i3(i3)
declare i4  @llvm.bitreverse.i4(i4)
declare i8  @llvm.bitreverse.i8(i8)
declare i32 @llvm.bitreverse.i32(i32)
declare <4 x i4>  @llvm.bitreverse.v4i4(<4 x i4>)
declare <4 x i32> @llvm.bitreverse.v4i32(<4 x i32>)
