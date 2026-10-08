; RUN: triton-llvm-opt -guard-masked-div-rem %s | FileCheck %s

define void @phi_div_of_zero_okay(i8 noundef %x, i8 %i, ptr %v) {
; CHECK-LABEL: @phi_div_of_zero_okay(
entry:
  %cmp = icmp ult i8 %i, 9
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %y = load i8, ptr %v, align 8
  br label %if.end

if.end:
  %yy = phi i8 [ %y, %if.then ], [ 0, %entry ]
  ; CHECK: [[FR:%.*]] = freeze i8 %yy
  ; CHECK-NEXT: [[CMP:%.*]] = icmp eq i8 [[FR]], 0
  ; CHECK-NEXT: [[SAFE:%.*]] = select i1 [[CMP]], i8 1, i8 [[FR]]
  ; CHECK-NEXT: %z = sdiv i8 %x, [[SAFE]]
  %z = sdiv i8 %x, %yy
  br i1 %cmp, label %if2.then, label %if2.end

if2.then:
  store i8 %z, ptr %v, align 8
  br label %if2.end

if2.end:
  ret void
}

define void @two_phi_div_of_zero_okay(i8 noundef %x, i8 %i, ptr %v) {
; CHECK-LABEL: @two_phi_div_of_zero_okay(
entry:
  %cmp = icmp ult i8 %i, 9
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %y = load i8, ptr %v, align 8
  %vv = getelementptr inbounds i64, ptr %v, i64 1
  %b = load i8, ptr %vv, align 8
  br label %if.end

if.end:
  %bb = phi i8 [ %b, %if.then ], [ undef, %entry ]
  %yy = phi i8 [ %y, %if.then ], [ 0, %entry ]
  ; CHECK: [[FR0:%.*]] = freeze i8 %yy
  ; CHECK-NEXT: [[CMP0:%.*]] = icmp eq i8 [[FR0]], 0
  ; CHECK-NEXT: [[SAFE0:%.*]] = select i1 [[CMP0]], i8 1, i8 [[FR0]]
  ; CHECK-NEXT: %z = sdiv i8 %x, [[SAFE0]]
  %z = sdiv i8 %x, %yy
  ; CHECK: [[FR1:%.*]] = freeze i8 %bb
  ; CHECK-NEXT: [[CMP1:%.*]] = icmp eq i8 [[FR1]], 0
  ; CHECK-NEXT: [[SAFE1:%.*]] = select i1 [[CMP1]], i8 1, i8 [[FR1]]
  ; CHECK-NEXT: %zz = sdiv i8 %x, [[SAFE1]]
  %zz = sdiv i8 %x, %bb
  br i1 %cmp, label %if2.then, label %if2.end

if2.then:
  store i8 %z, ptr %v, align 8
  br label %if2.end

if2.end:
  ret void
}

; A vectorized masked load (sizePerThread=2): the zero default reaches the divisor through extractelement, and the
; division sits in a later block. The phi-shape guard missed this; IGC's scalarizer then exposes it to SimplifyCFG.
define void @vector_phi_extract_later_block(i64 %x, i1 %mask, i1 %c2, ptr %v, ptr %out) {
; CHECK-LABEL: @vector_phi_extract_later_block(
entry:
  br i1 %mask, label %load, label %merge

load:
  %l = load <2 x i64>, ptr %v, align 16
  br label %merge

merge:
  %p = phi <2 x i64> [ %l, %load ], [ zeroinitializer, %entry ]
  %e = extractelement <2 x i64> %p, i64 0
  br i1 %c2, label %other, label %use

other:
  br label %use

use:
  ; CHECK: [[FR:%.*]] = freeze i64 %e
  ; CHECK-NEXT: [[CMP:%.*]] = icmp eq i64 [[FR]], 0
  ; CHECK-NEXT: [[SAFE:%.*]] = select i1 [[CMP]], i64 1, i64 [[FR]]
  ; CHECK-NEXT: %r = srem i64 %x, [[SAFE]]
  %r = srem i64 %x, %e
  store i64 %r, ptr %out, align 8
  ret void
}

define <2 x i32> @vector_divisor(<2 x i32> %x, <2 x i32> %d) {
; CHECK-LABEL: @vector_divisor(
; CHECK: [[FR:%.*]] = freeze <2 x i32> %d
; CHECK-NEXT: [[CMP:%.*]] = icmp eq <2 x i32> [[FR]], zeroinitializer
; CHECK-NEXT: [[SAFE:%.*]] = select <2 x i1> [[CMP]], <2 x i32> {{.*}}1{{.*}}, <2 x i32> [[FR]]
; CHECK-NEXT: %r = udiv <2 x i32> %x, [[SAFE]]
  %r = udiv <2 x i32> %x, %d
  ret <2 x i32> %r
}

declare spir_func i64 @_Z27__spirv_PredicatedLoadINTELPU3AS1lbl(ptr addrspace(1), i1, i64)

; Predicated loads hide the zero default behind a call, but the division by zero still executes on masked-off lanes.
define i64 @predicated_load_divisor(i64 %x, ptr addrspace(1) %ptr, i1 %mask) {
; CHECK-LABEL: @predicated_load_divisor(
; CHECK: [[FR:%.*]] = freeze i64 %d
; CHECK-NEXT: [[CMP:%.*]] = icmp eq i64 [[FR]], 0
; CHECK-NEXT: [[SAFE:%.*]] = select i1 [[CMP]], i64 1, i64 [[FR]]
; CHECK-NEXT: %r = sdiv i64 %x, [[SAFE]]
  %d = call spir_func i64 @_Z27__spirv_PredicatedLoadINTELPU3AS1lbl(ptr addrspace(1) %ptr, i1 %mask, i64 0)
  %r = sdiv i64 %x, %d
  ret i64 %r
}

; Divisors that are provably non-zero are left alone.
define i32 @const_nonzero_divisor(i32 %x) {
; CHECK-LABEL: @const_nonzero_divisor(
; CHECK-NOT: select
; CHECK: %r = sdiv i32 %x, 7
  %r = sdiv i32 %x, 7
  ret i32 %r
}

define i32 @known_nonzero_divisor(i32 %x, i32 %y) {
; CHECK-LABEL: @known_nonzero_divisor(
; CHECK-NOT: select
; CHECK: %r = urem i32 %x, %d
  %d = or i32 %y, 1
  %r = urem i32 %x, %d
  ret i32 %r
}
