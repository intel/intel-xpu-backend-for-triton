; RUN: triton-llvm-opt -scalarize-ptr-vectors %s | FileCheck %s

; The pass turns dynamic indexing of `<N x ptr>` vectors into select cascades
; that the SPIR-V translator can handle.

; Dynamic index into 2 pointers -> one select.
; CHECK-LABEL: @dynamic_index_two
define ptr addrspace(3) @dynamic_index_two(ptr addrspace(3) %b0, ptr addrspace(3) %b1, i32 %idx) {
  ; CHECK:      %[[CMP:.*]] = icmp eq i32 %idx, 0
  ; CHECK-NEXT: %[[SEL:.*]] = select i1 %[[CMP]], ptr addrspace(3) %b0, ptr addrspace(3) %b1
  ; CHECK-NEXT: ret ptr addrspace(3) %[[SEL]]
  ; CHECK-NOT:  extractelement
  %v0 = insertelement <2 x ptr addrspace(3)> undef, ptr addrspace(3) %b0, i32 0
  %v1 = insertelement <2 x ptr addrspace(3)> %v0, ptr addrspace(3) %b1, i32 1
  %p = extractelement <2 x ptr addrspace(3)> %v1, i32 %idx
  ret ptr addrspace(3) %p
}

; Dynamic index into 4 pointers -> chained selects, last index first.
; CHECK-LABEL: @dynamic_index_four
define ptr addrspace(3) @dynamic_index_four(ptr addrspace(3) %b0, ptr addrspace(3) %b1, ptr addrspace(3) %b2, ptr addrspace(3) %b3, i32 %idx) {
  ; CHECK:      %[[CMP2:.*]] = icmp eq i32 %idx, 2
  ; CHECK-NEXT: %[[SEL2:.*]] = select i1 %[[CMP2]], ptr addrspace(3) %b2, ptr addrspace(3) %b3
  ; CHECK-NEXT: %[[CMP1:.*]] = icmp eq i32 %idx, 1
  ; CHECK-NEXT: %[[SEL1:.*]] = select i1 %[[CMP1]], ptr addrspace(3) %b1, ptr addrspace(3) %[[SEL2]]
  ; CHECK-NEXT: %[[CMP0:.*]] = icmp eq i32 %idx, 0
  ; CHECK-NEXT: %[[SEL0:.*]] = select i1 %[[CMP0]], ptr addrspace(3) %b0, ptr addrspace(3) %[[SEL1]]
  ; CHECK-NEXT: ret ptr addrspace(3) %[[SEL0]]
  ; CHECK-NOT:  extractelement
  %v0 = insertelement <4 x ptr addrspace(3)> undef, ptr addrspace(3) %b0, i32 0
  %v1 = insertelement <4 x ptr addrspace(3)> %v0, ptr addrspace(3) %b1, i32 1
  %v2 = insertelement <4 x ptr addrspace(3)> %v1, ptr addrspace(3) %b2, i32 2
  %v3 = insertelement <4 x ptr addrspace(3)> %v2, ptr addrspace(3) %b3, i32 3
  %p = extractelement <4 x ptr addrspace(3)> %v3, i32 %idx
  ret ptr addrspace(3) %p
}

; Constant index -> just the matching pointer, no select.
; CHECK-LABEL: @constant_index
define ptr addrspace(3) @constant_index(ptr addrspace(3) %b0, ptr addrspace(3) %b1) {
  ; CHECK-NOT: extractelement
  ; CHECK-NOT: select
  ; CHECK:     ret ptr addrspace(3) %b1
  %v0 = insertelement <2 x ptr addrspace(3)> undef, ptr addrspace(3) %b0, i32 0
  %v1 = insertelement <2 x ptr addrspace(3)> %v0, ptr addrspace(3) %b1, i32 1
  %p = extractelement <2 x ptr addrspace(3)> %v1, i32 1
  ret ptr addrspace(3) %p
}

; Out-of-bounds constant index -> poison, and still erased (no leftover use).
; CHECK-LABEL: @out_of_bounds_constant_index
define ptr addrspace(3) @out_of_bounds_constant_index(ptr addrspace(3) %b0, ptr addrspace(3) %b1) {
  ; CHECK-NOT: extractelement
  ; CHECK:     ret ptr addrspace(3) poison
  %v0 = insertelement <2 x ptr addrspace(3)> undef, ptr addrspace(3) %b0, i32 0
  %v1 = insertelement <2 x ptr addrspace(3)> %v0, ptr addrspace(3) %b1, i32 1
  %p = extractelement <2 x ptr addrspace(3)> %v1, i32 5
  ret ptr addrspace(3) %p
}

; Non-pointer vectors are left alone.
; CHECK-LABEL: @non_ptr_vector_untouched
define i32 @non_ptr_vector_untouched(i32 %a, i32 %b, i32 %idx) {
  ; CHECK: insertelement
  ; CHECK: extractelement
  %v0 = insertelement <2 x i32> undef, i32 %a, i32 0
  %v1 = insertelement <2 x i32> %v0, i32 %b, i32 1
  %e = extractelement <2 x i32> %v1, i32 %idx
  ret i32 %e
}
