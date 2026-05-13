; RUN: triton-llvm-opt -expand-subbyte-bitwise-and %s | FileCheck %s

; CHECK-LABEL: @and_i4
define i4 @and_i4(i4 %a, i4 %b) {
; CHECK-NOT: and i4
; CHECK:     [[A32:%.*]] = zext i4 %a to i32
; CHECK:     [[B32:%.*]] = zext i4 %b to i32
; CHECK:     [[R32:%.*]] = and i32 [[A32]], [[B32]]
; CHECK:     [[RES:%.*]] = trunc i32 [[R32]] to i4
; CHECK:     ret i4 [[RES]]
  %r = and i4 %a, %b
  ret i4 %r
}

; CHECK-LABEL: @and_i3
define i3 @and_i3(i3 %a, i3 %b) {
; CHECK-NOT: and i3
; CHECK:     [[A32:%.*]] = zext i3 %a to i32
; CHECK:     [[B32:%.*]] = zext i3 %b to i32
; CHECK:     [[R32:%.*]] = and i32 [[A32]], [[B32]]
; CHECK:     [[RES:%.*]] = trunc i32 [[R32]] to i3
; CHECK:     ret i3 [[RES]]
  %r = and i3 %a, %b
  ret i3 %r
}

; >=8-bit `and` is not touched.
; CHECK-LABEL: @and_i8
define i8 @and_i8(i8 %a, i8 %b) {
; CHECK: and i8 %a, %b
  %r = and i8 %a, %b
  ret i8 %r
}
