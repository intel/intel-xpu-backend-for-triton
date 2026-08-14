// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s

// Test: loop with two dots sharing operand A is distributed into two loops.
// Each loop loads the shared A from %arg0 and only its own B operand. The
// two scf.for results are wired back to the original op's results: result 0
// comes from the first (dot0) loop, result 1 from the second (dot1) loop.
// CHECK-LABEL: @dual_dot_distribute
// CHECK: %[[LOOP1:.*]]:2 = scf.for
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK-NOT: tt.descriptor_load %arg2
// CHECK:   tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: %[[LOOP2:.*]]:2 = scf.for
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK-NOT: tt.descriptor_load %arg1
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1
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

// -----

// Test: dot0's B operand depends on dot0's OWN accumulator (%acc_g) via a
// pure arith.addf. This is safe -- unlike a dependency on the OTHER dot's
// accumulator (which would read a frozen/dead pass-through slot in the
// split loop), %acc_g evolves normally within the same new loop that
// computes dot0, so the value read is identical to the original fused loop.
// The loop must still be fully distributed. The accumulator iter_arg name is
// captured from the loop header instead of hard-coded, since MLIR restarts
// block-argument numbering per region: the SAME name (e.g. %arg3) prints as
// the live accumulator in loop1, the frozen one in loop2, and also in the
// un-distributed printing of this function, so a literal match would not
// distinguish them.
// CHECK-LABEL: @own_accumulator_dependency_distribute
// CHECK: %[[LOOP1:.*]]:2 = scf.for {{.*}}iter_args(%[[ACC1:[^ ]+]] = %cst
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WRAW1:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[WG:.*]] = arith.addf %[[WRAW1]], %[[ACC1]]
// CHECK:   tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: %[[LOOP2:.*]]:2 = scf.for
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WRAW2:.*]] = tt.descriptor_load %arg1
// The arith.addf belongs to dot0's operand slice only, so it must NOT be cloned
// into loop2 (where %acc_g is the frozen pass-through slot).
// CHECK-NOT: arith.addf
// CHECK:   tt.dot %[[X2]], %[[WRAW2]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @own_accumulator_dependency_distribute(%arg0: !tt.tensordesc<128x128xf32>, %arg1: !tt.tensordesc<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c128_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
      %w_raw = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
      // dot0's B operand depends on dot0's OWN accumulator -- safe, since
      // that accumulator evolves normally in the loop that computes dot0.
      %wg = arith.addf %w_raw, %acc_g : tensor<128x128xf32>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %w_raw, %acc_fc, inputPrecision = tf32 : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
