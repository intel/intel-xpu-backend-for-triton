// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" 2>&1 | FileCheck %s --check-prefix=STATS

// COM: Statistics for the shared-source phase of HoistLayoutConversions (issue
// COM: #8202). Each candidate must be counted exactly once however many times
// COM: it is weighed: as hoisted if either phase takes it, else under the
// COM: reason the one-at-a-time phase first refused it for. The counters are
// COM: process-global and printed once per -split-input-file section, so each
// COM: section pins the running totals. A STATS-NOT follows every section's
// COM: line, so each directive must match the next statistics line printed
// COM: and none can skip ahead to a later section's. The IR is copied from
// COM: cases 16, 45 and 46 of hoist-layout-conversions.mlir, which explain the
// COM: byte math; this file runs at grf-mode=default only.

// COM: Case 16's shape: loop 2's conversion refused alone by the peak, loop
// COM: 1's credited past that refused twin and hoisted alone, then loop 2's
// COM: overturned by the singleton group. Two considered, two hoisted,
// COM: nothing rejected.
// STATS: [HoistLayoutConversions] considered=2 hoisted=2 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=0 skipped_other=0
// STATS-NOT: [HoistLayoutConversions]


#blocked16 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas16 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a16 = #ttg.dot_op<{opIdx = 0, parent = #dpas16, kWidth = 1}>
#dot_b16 = #ttg.dot_op<{opIdx = 1, parent = #dpas16, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @shared_source_two_sibling_loops(%arg0: tensor<128x16xf16, #blocked16>, %argB: tensor<16x16xf16, #dot_b16>,
                                           %arg2: tensor<128x16xf32, #dpas16>) -> tensor<128x16xf32, #dpas16> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked16>
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas16>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked16> -> tensor<128x16xf16, #dot_a16>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas16>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas16>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas16>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas16>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas16>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas16>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a16> * tensor<16x16xf16, #dot_b16> -> tensor<128x16xf32, #dpas16>
      scf.yield %d1 : tensor<128x16xf32, #dpas16>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas16>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked16> -> tensor<128x16xf16, #dot_a16>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a16> * tensor<16x16xf16, #dot_b16> -> tensor<128x16xf32, #dpas16>
      scf.yield %d2 : tensor<128x16xf32, #dpas16>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas16>
  }
}

// -----

// COM: Case 45's shape: loops 6..2 are refused alone by the peak (five
// COM: candidates). Loop 1, decided last, is credited as if %src were fully
// COM: retired (loops 6..2's refusals share %src's exact (source, result
// COM: type), so hoisting %cvt1 would dominate and retire them too -- see
// COM: hoistRetiresSource's doc comment), so it clears both gates and is
// COM: hoisted directly. The group of five is then refused by the joint
// COM: trial's measured peak regardless, same as before this credit fix,
// COM: but %cvt1 is no longer part of that group (it already hoisted): +6
// COM: considered, +1 hoisted, +5 under whichever peak counter loops 6..2's
// COM: projections reported.
// STATS: [HoistLayoutConversions] considered=8 hoisted=3 rejected_pressure=0 rejected_function_peak_exact=5 rejected_function_peak_fallback=0 skipped_other=0
// STATS-NOT: [HoistLayoutConversions]


#blocked45 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas45 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a45 = #ttg.dot_op<{opIdx = 0, parent = #dpas45, kWidth = 1}>
#dot_b45 = #ttg.dot_op<{opIdx = 1, parent = #dpas45, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @shared_source_joint_over_peak(%arg0: tensor<128x16xf16, #blocked45>, %argB: tensor<16x16xf16, #dot_b45>,
      %arg2: tensor<128x16xf32, #dpas45>) -> tensor<128x16xf32, #dpas45> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked45>
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas45>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas45>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas45>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas45>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas45>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas45>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d1 : tensor<128x16xf32, #dpas45>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d2 : tensor<128x16xf32, #dpas45>
    }
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt3 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d3 = tt.dot %cvt3, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d3 : tensor<128x16xf32, #dpas45>
    }
    %r4 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r3) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt4 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d4 = tt.dot %cvt4, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d4 : tensor<128x16xf32, #dpas45>
    }
    %r5 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r4) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt5 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d5 = tt.dot %cvt5, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d5 : tensor<128x16xf32, #dpas45>
    }
    %r6 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r5) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt6 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d6 = tt.dot %cvt6, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d6 : tensor<128x16xf32, #dpas45>
    }
    tt.return %r6 : tensor<128x16xf32, #dpas45>
  }
}

// -----

// COM: Case 46's shape: both refused alone by the loop-level gate, and the
// COM: group refused by it again: +2 considered, +2 rejected_pressure.
// STATS: [HoistLayoutConversions] considered=10 hoisted=3 rejected_pressure=2 rejected_function_peak_exact=5 rejected_function_peak_fallback=0 skipped_other=0
// STATS-NOT: [HoistLayoutConversions]


#blocked46 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas46 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a46 = #ttg.dot_op<{opIdx = 0, parent = #dpas46, kWidth = 1}>
#dot_b46 = #ttg.dot_op<{opIdx = 1, parent = #dpas46, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @shared_source_joint_over_loop_budget(%arg0: tensor<128x16xf16, #blocked46>, %argB: tensor<16x16xf16, #dot_b46>,
      %arg2: tensor<128x16xf32, #dpas46>) -> tensor<128x16xf32, #dpas46> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %p1 = arith.addf %arg2, %arg2 : tensor<128x16xf32, #dpas46>
    %p2 = arith.addf %arg2, %p1 : tensor<128x16xf32, #dpas46>
    %p3 = arith.addf %arg2, %p2 : tensor<128x16xf32, #dpas46>
    %p4 = arith.addf %p3, %p1 : tensor<128x16xf32, #dpas46>
    %p5 = arith.addf %arg2, %p2 : tensor<128x16xf32, #dpas46>
    %p6 = arith.addf %p4, %p5 : tensor<128x16xf32, #dpas46>
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked46>
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %p6) -> (tensor<128x16xf32, #dpas46>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked46> -> tensor<128x16xf16, #dot_a46>
      %d1 = tt.dot %cvt1, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a46> * tensor<16x16xf16, #dot_b46> -> tensor<128x16xf32, #dpas46>
      scf.yield %d1 : tensor<128x16xf32, #dpas46>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas46>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked46> -> tensor<128x16xf16, #dot_a46>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a46> * tensor<16x16xf16, #dot_b46> -> tensor<128x16xf32, #dpas46>
      scf.yield %d2 : tensor<128x16xf32, #dpas46>
    }
    %keep = arith.addf %src, %src : tensor<128x16xf16, #blocked46>
    tt.return %r2 : tensor<128x16xf32, #dpas46>
  }
}
