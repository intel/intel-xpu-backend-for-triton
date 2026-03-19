// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

tt.func public @fuseLoadWithReshape1(%arg0: tensor<256x32xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c64_i32, %c1024_i32], [%c1024_i64, %c4_i64, %c1_i64] : <bf16>, <tensor<1x32x256xbf16>>
  %3 = tt.descriptor_load %0[%c2_i32, %c1_i32, %c0_i32]  : !tt.tensordesc<tensor<1x32x256xbf16>> -> tensor<1x32x256xbf16>
  %4 = tt.reshape %3 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
  %5 = tt.dot %arg0, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape1
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c4_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK: [[MUL1:%.*]] = arith.muli %c1_i32, [[TRUNC]] : i32
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c64_i32 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[ADD1]], %c1024_i32], [%c4_i64, %c1_i64] : <bf16>, <tensor<32x256xbf16>>
// CHECK: [[MUL2:%.*]] = arith.muli %c2_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %c1_i32 : i32
// CHECK: [[LOAD_B:%.*]] = tt.descriptor_load [[DESC]][[[ADD2]], %c0_i32] : !tt.tensordesc<tensor<32x256xbf16>> -> tensor<32x256xbf16>
// CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

tt.func public @fuseLoadWithReshape2(%arg0: tensor<32x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c512_i32, %c1024_i32, %c32_i32], [%c1024_i64, %c1_i64, %c512_i64]: <bf16>, <tensor<1x256x32xbf16>>
  %res:2 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32>, i32) : i32 {
    %1 = tt.descriptor_load %0[%c32_i32, %c32_i32, %c0_i32] : !tt.tensordesc<tensor<1x256x32xbf16>> -> tensor<1x256x32xbf16>
    %2 = tt.reshape %1 : tensor<1x256x32xbf16> -> tensor<256x32xbf16>
    %4 = tt.dot %2, %arg0, %arg4, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    %5 = arith.addi %arg5, %c32_i32 : i32
    scf.yield %4, %5 : tensor<256x256xf32>, i32
  }
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape2
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c1_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK: [[MUL1:%.*]] = arith.muli %c512_i32, [[TRUNC]] : i32
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c1024_i32 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[ADD1]], %c32_i32], [%c1_i64, %c512_i64] : <bf16>, <tensor<256x32xbf16>>
// CHECK: scf.for
// CHECK:   [[MUL2:%.*]] = arith.muli %c32_i32, [[TRUNC]] : i32
// CHECK:   [[ADD2:%.*]] = arith.addi [[MUL2]], %c32_i32 : i32
// CHECK:   [[LOAD_A:%.*]] = tt.descriptor_load [[DESC]][[[ADD2]], %c0_i32] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
// CHECK:   tt.dot [[LOAD_A]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
