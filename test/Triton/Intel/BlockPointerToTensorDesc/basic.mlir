// RUN: triton-opt %s -triton-intel-block-pointer-to-tdesc | FileCheck %s

// CHECK-NOT: tt.make_tensor_ptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.advance
tt.func public @load(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @load
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32)
  // CHECK-DAG:     [[shape0:%.*]] = arith.trunci [[shape0_i64]] : i64 to i32
  // CHECK-DAG:     [[shape1:%.*]] = arith.trunci [[shape1_i64]] : i64 to i32
  // CHECK:         [[tdesc:%.*]] = tt.make_tensor_descriptor [[ptr]], [[[shape0]], [[shape1]]], [[[stride0]], [[stride1]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK:         tt.descriptor_load [[tdesc]][[[offset0]], [[offset1]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}
