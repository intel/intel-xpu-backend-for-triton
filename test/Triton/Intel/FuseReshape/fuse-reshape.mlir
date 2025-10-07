// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: tt.load -> tt.reshape -> tt.dot chain, not in a loop.
  tt.func public @fuseLoadWithReshape1(%arg0: !tt.ptr<tensor<256x32xbf16>>, %arg1: !tt.ptr<bf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32    
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %0 = tt.make_tensor_ptr %arg1, [%c2_i64, %c1_i64, %c1024_i64], [%c3_i64, %c1024_i64, %c1_i64], [%c2_i32, %c1_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x32x256xbf16>>
    %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
    %3 = tt.load %0 {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x32x256xbf16>>
    %4 = tt.reshape %3 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
    %5 = tt.dot %1, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    tt.return
  }
  // CHECK-LABEL: fuseLoadWithReshape1
  // CHECK-NOT: tt.reshape
  // CHECK: [[TRUNC:%.*]] = arith.trunci %c3_i64 : i64 to i32
  // CHECK: [[MUL:%.*]] = arith.muli [[TRUNC]], %c2_i32 : i32
  // CHECK: [[ADD:%.*]] = arith.addi [[MUL]], %c0_i32 : i32
  // CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [%c1_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c1_i32, [[ADD]]] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK: [[LOAD_B:%.*]] = tt.load [[PTR]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
  // CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
}
