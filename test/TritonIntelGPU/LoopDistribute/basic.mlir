// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s

// Test: loop with two dots sharing operand A is distributed into two loops.
// Each loop loads the shared A from %arg0 and only its own B operand.
// CHECK-LABEL: @dual_dot_distribute
// CHECK: scf.for
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK-NOT: tt.descriptor_load %arg2
// CHECK:   tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: scf.for
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK-NOT: tt.descriptor_load %arg1
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @dual_dot_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
