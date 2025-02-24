; RUN: triton-llvm-opt -freeze-masked-div-rem %s | FileCheck %s

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
  ; CHECK: [[F0:%.*]] = freeze i8 %yy
  ; CHECK-NEXT: %z = sdiv i8 %x, [[F0:%.*]]
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
  ; CHECK: [[F0:%.*]] = freeze i8 %yy
  ; CHECK-NEXT: %z = sdiv i8 %x, [[F0:%.*]]
  %z = sdiv i8 %x, %yy
  ; CHECK: [[F1:%.*]] = freeze i8 %bb
  ; CHECK-NEXT: %zz = sdiv i8 %x, [[F1:%.*]]
  %zz = sdiv i8 %x, %bb
  br i1 %cmp, label %if2.then, label %if2.end

if2.then:
  store i8 %z, ptr %v, align 8
  br label %if2.end

if2.end:
  ret void
}
