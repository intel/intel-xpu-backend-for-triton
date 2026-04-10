; RUN: triton-llvm-opt -expand-sadd-overflow %s | FileCheck %s

; CHECK-LABEL: @sadd_i32
define {i32, i1} @sadd_i32(i32 %a, i32 %b) {
; CHECK-NOT: @llvm.sadd.with.overflow
; CHECK:     [[SUM:%.*]] = add i32 %a, %b
; CHECK:     [[XOR_AB:%.*]] = xor i32 %a, %b
; CHECK:     [[NOT:%.*]] = xor i32 [[XOR_AB]], -1
; CHECK:     [[XOR_AS:%.*]] = xor i32 %a, [[SUM]]
; CHECK:     [[AND:%.*]] = and i32 [[NOT]], [[XOR_AS]]
; CHECK:     [[OV:%.*]] = icmp slt i32 [[AND]], 0
; CHECK:     [[R0:%.*]] = insertvalue { i32, i1 } undef, i32 [[SUM]], 0
; CHECK:     [[R1:%.*]] = insertvalue { i32, i1 } [[R0]], i1 [[OV]], 1
; CHECK:     ret { i32, i1 } [[R1]]
  %r = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  ret {i32, i1} %r
}

; CHECK-LABEL: @sadd_i16
define {i16, i1} @sadd_i16(i16 %a, i16 %b) {
; CHECK-NOT: @llvm.sadd.with.overflow
; CHECK:     [[SUM:%.*]] = add i16 %a, %b
; CHECK:     icmp slt i16 {{%.*}}, 0
; CHECK:     ret { i16, i1 }
  %r = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
  ret {i16, i1} %r
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i16, i1} @llvm.sadd.with.overflow.i16(i16, i16)
