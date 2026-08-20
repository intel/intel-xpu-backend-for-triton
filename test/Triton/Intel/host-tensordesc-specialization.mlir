// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s

// Test that shape divisibility attributes from specialization are correctly
// propagated to tt.divisibility on shape block arguments.
// Frontend flattened layout: descriptor, shape[0..rank-1], stride[0..rank-1]
module {
  tt.func public @shape_divisibility(
      %arg0: !tt.tensordesc<128x64xf16>
             {tt.shape.0.divisibility = 128 : i32, tt.shape.1.divisibility = 64 : i32},
      %arg1: i32, %arg2: i32,
      %arg3: i64, %arg4: i64) -> tensor<128x64xf16> {
    %c0 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
    tt.return %0 : tensor<128x64xf16>
  }
}

// CHECK-LABEL: @shape_divisibility
// Descriptor expands to (ptr, 2*rank i64 args, 2 i1 args).
// CHECK-SAME: %[[PTR:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK-SAME: %[[SHAPE0:[^:]*]]: i64
// CHECK-SAME: %[[SHAPE1:[^:]*]]: i64
// CHECK-SAME: %[[STRIDE0:[^:]*]]: i64
// CHECK-SAME: %[[STRIDE1:[^:]*]]: i64
// CHECK-SAME: %[[PAD:[^:]*]]: i1
// CHECK-SAME: %[[ROUND:[^:]*]]: i1
// Frontend shape args (i32) become block arguments shifted by descriptor expansion.
// CHECK-SAME: %[[ORIG_SHAPE0:[^:]*]]: i32 {tt.divisibility = 128 : i32}
// CHECK-SAME: %[[ORIG_SHAPE1:[^:]*]]: i32 {tt.divisibility = 64 : i32}
// Frontend stride args (i64) follow shape args.
// CHECK-SAME: %[[ORIG_STRIDE0:[^:]*]]: i64 {tt.divisibility = 8 : i32}
// CHECK-SAME: %[[ORIG_STRIDE1:[^:]*]]: i64
// CHECK: %[[C1:.*]] = arith.constant 1 : i64
// CHECK: tt.make_tensor_descriptor %[[PTR]], [%[[ORIG_SHAPE0]], %[[ORIG_SHAPE1]]], [%[[ORIG_STRIDE0]], %[[C1]]]

// -----

// Test that NaN padding attribute is correctly handled.
module {
  tt.func public @padding(
      %arg0: !tt.tensordesc<128x64xf16> {tt.padding = 1 : i32},
      %arg1: i32, %arg2: i32,
      %arg3: i64, %arg4: i64) -> tensor<128x64xf16> {
    %c0 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
    tt.return %0 : tensor<128x64xf16>
  }
}

// CHECK-LABEL: @padding
// CHECK: tt.make_tensor_descriptor %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}] {padding = 2 : i32}

// -----

// Test that base pointer divisibility is always 16 bytes.
module {
  tt.func public @base_ptr_divisibility(
      %arg0: !tt.tensordesc<128x64xf16>,
      %arg1: i32, %arg2: i32,
      %arg3: i64, %arg4: i64) -> tensor<128x64xf16> {
    %c0 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
    tt.return %0 : tensor<128x64xf16>
  }
}

// CHECK-LABEL: @base_ptr_divisibility
// Base pointer always gets 16-byte divisibility.
// CHECK-SAME: %[[PTR:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
