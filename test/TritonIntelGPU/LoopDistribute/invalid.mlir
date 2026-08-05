// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s

// Test: loop with a single dot is NOT distributed (not exactly 2 dots).
// CHECK-LABEL: @single_dot_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @single_dot_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>) -> tensor<128x128xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %w = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d = tt.dot %x, %w, %acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d : tensor<128x128xf32>
    }
    tt.return %0 : tensor<128x128xf32>
  }
}

// -----

// Test: loop with three dots is NOT distributed (not exactly 2 dots).
// CHECK-LABEL: @three_dots_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @three_dots_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>, %arg3: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%a0 = %cst, %a1 = %cst, %a2 = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %w0 = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %w1 = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %w2 = tt.descriptor_load %arg3[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %w0, %a0, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %w1, %a1, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d2 = tt.dot %x, %w2, %a2, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1, %d2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: dot accumulator not a direct iter_arg is NOT distributed.
// CHECK-LABEL: @non_direct_acc_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @non_direct_acc_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      // acc_g is modified before being used as accumulator
      %modified_acc = arith.addf %acc_g, %cst1 : tensor<128x128xf32>
      %d0 = tt.dot %x, %wg, %modified_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: dot result not directly yielded is NOT distributed.
// CHECK-LABEL: @dot_result_not_yielded_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @dot_result_not_yielded_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // yield a modified version of d0 instead of d0 directly
      %modified = arith.addf %d0, %cst1 : tensor<128x128xf32>
      scf.yield %modified, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: loop with unclassified side-effecting op is NOT distributed.
// CHECK-LABEL: @side_effecting_op_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @side_effecting_op_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>, %arg3: !tt.tensordesc<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
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
      // Side-effecting store not in either dot's slice
      tt.descriptor_store %arg3[%c0_i32, %c0_i32], %d0 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: dots with inter-dependencies are NOT distributed.
// dot1's A operand is derived from dot0's result, creating a dependency.
// CHECK-LABEL: @inter_dependent_dots_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @inter_dependent_dots_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<128x64xbf16>) -> (tensor<128x128xf32>, tensor<128x64xf32>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc0 = %cst0, %acc1 = %cst1) -> (tensor<128x128xf32>, tensor<128x64xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %w = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %w, %acc0, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // d1's A operand is derived from d0's result
      %trunc = arith.truncf %d0 : tensor<128x128xf32> to tensor<128x128xbf16>
      %x2 = tt.descriptor_load %arg2[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %d1 = tt.dot %trunc, %x2, %acc1, inputPrecision = tf32 : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x64xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x64xf32>
  }
}
