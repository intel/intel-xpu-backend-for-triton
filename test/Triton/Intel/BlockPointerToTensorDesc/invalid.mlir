// RUN: triton-opt %s -triton-intel-block-pointer-to-tdesc | FileCheck %s

// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load

tt.func public @no_boundary_check(%ptr: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @no_boundary_check
  %0 = tt.make_tensor_ptr %ptr, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.load %0 : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}

tt.func public @only_boundary_check_0(%ptr: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @only_boundary_check_0
  %0 = tt.make_tensor_ptr %ptr, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}

tt.func public @only_boundary_check_1(%ptr: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @only_boundary_check_1
  %0 = tt.make_tensor_ptr %ptr, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}

// COM: Current limitation: it can be extended to return offsets in addition to the descriptors from scf.if in the future.
tt.func public @if(%ptr: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32, %cond: i1) {
  // CHECK-LABEL: tt.func public @if
  %0 = scf.if %cond -> (!tt.ptr<tensor<256x32xbf16>>) {
    %1 = tt.make_tensor_ptr %ptr, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
    scf.yield %1 : !tt.ptr<tensor<256x32xbf16>>
  } else {
    %2 = tt.make_tensor_ptr %ptr, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
    scf.yield %2 : !tt.ptr<tensor<256x32xbf16>>
  }
  %3 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}
