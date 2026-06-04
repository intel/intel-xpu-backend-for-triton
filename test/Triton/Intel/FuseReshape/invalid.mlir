// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// CHECK-LABEL: noFuseWithoutBlockIO
// CHECK: tt.descriptor_load
// CHECK: tt.reshape
tt.func public @noFuseWithoutBlockIO(%arg0: tensor<256x32xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c64_i32, %c1024_i32], [%c1024_i64, %c4_i64, %c1_i64] : <bf16>, <1x32x256xbf16>
  %3 = tt.descriptor_load %0[%c2_i32, %c1_i32, %c0_i32] : !tt.tensordesc<1x32x256xbf16> -> tensor<1x32x256xbf16>
  %4 = tt.reshape %3 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
  %5 = tt.dot %arg0, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
  tt.return
}
