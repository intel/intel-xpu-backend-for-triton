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

// -----

// Test: a side-effecting op (tt.atomic_rmw) sits inside what would otherwise
// be dot0's backward slice (its result feeds into computing dot0's B
// operand), rather than floating free outside both slices. This must still
// be rejected -- slice members must be side-effect free to be safely
// replicated into both new loops.
// CHECK-LABEL: @side_effecting_op_in_slice_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @side_effecting_op_in_slice_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>, %rmw_ptr: tensor<64x128x!tt.ptr<f32>>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cstf = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg_raw = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      // Side-effecting atomic feeds into computing dot0's B operand.
      %old = tt.atomic_rmw fadd, acq_rel, gpu, %rmw_ptr, %cstf : (tensor<64x128x!tt.ptr<f32>>, tensor<64x128xf32>) -> tensor<64x128xf32>
      %old_bf16 = arith.truncf %old : tensor<64x128xf32> to tensor<64x128xbf16>
      %wg = arith.addf %wg_raw, %old_bf16 : tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a loop-carried chain (%carry) is updated from a dot accumulator's
// block argument (%acc_g) directly, rather than from a value that is safe to
// replicate identically into both new loops. Replicating this chain would
// read a different (frozen or partial) %acc_g in each new loop, so
// distribution must be rejected.
// CHECK-LABEL: @carried_depends_on_acc_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_depends_on_acc_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %carry's update reads dot0's accumulator block-argument directly.
      %bad = arith.addf %carry, %acc_g : tensor<128x128xf32>
      scf.yield %d0, %d1, %bad : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a loop-carried chain (%carry) is updated from a dot's RESULT (%d0)
// directly, rather than the accumulator iter_arg. This must also be
// rejected -- a carried chain may not depend on either dot's output.
// CHECK-LABEL: @carried_depends_on_dot_result_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_depends_on_dot_result_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %carry's update derives from dot0's RESULT directly (not the
      // accumulator iter_arg).
      %bad = arith.addf %carry, %d0 : tensor<128x128xf32>
      scf.yield %d0, %d1, %bad : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a pure op that belongs to neither dot's slice nor any carried chain,
// and whose result is not used anywhere (not a dot operand, not any yield
// operand), is a genuinely unclassified op. It must be rejected. This only
// survives as a standalone test case because we bypass the canonicalizer
// that would normally dead-code-eliminate it in the real pipeline.
// CHECK-LABEL: @unclassified_dead_op_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @unclassified_dead_op_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
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
      // %dead is pure, but belongs to neither dot's slice nor any carried
      // chain, and has no further use.
      %dead = arith.addf %acc_g, %acc_g : tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
