// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" | FileCheck %s --check-prefixes=CHECK,GRF128
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128" | FileCheck %s --check-prefixes=CHECK,GRF128
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=256" | FileCheck %s --check-prefixes=CHECK,GRF256
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" -tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=MERGE

// COM: The MERGE run chains this pass with remove_layout_conversions, whose remat
// COM: cache replaces a refused "twin" (same source, same result type) with the
// COM: hoisted conversion that dominates it; the credit in hoistRetiresSource
// COM: relies on that. Only @shared_source_joint_over_peak and
// COM: @no_twin_credit_for_different_result_type carry MERGE checks.

// COM: The statistics counters are process-global and are printed once per
// COM: -split-input-file section, so the figures accumulate down the file. The
// COM: first directive pins the field set and its order against any output line;
// COM: the second pins the exact totals, and the STATS-NOT after it forbids any
// COM: later statistics line, so the totals must be on the final section's line
// COM: rather than on any earlier line that happens to carry the same figures.
// COM: Update the totals whenever a case is added or a verdict changes, and
// COM: check the sum: hoisted + the three rejected counters + skipped_other must
// COM: equal considered.
// STATS: [HoistLayoutConversions] considered={{[0-9]+}} hoisted={{[0-9]+}} rejected_pressure={{[0-9]+}} rejected_function_peak_exact={{[0-9]+}} rejected_function_peak_fallback={{[0-9]+}} skipped_other={{[0-9]+}}
// STATS: [HoistLayoutConversions] considered=95 hoisted=44 rejected_pressure=16 rejected_function_peak_exact=10 rejected_function_peak_fallback=16 skipped_other=9
// STATS-NOT: [HoistLayoutConversions]

// COM: Case 1: Hoist ConvertLayoutOp with DotOperandEncoding out of scf.for loop.
// COM: The source of the convert_layout is defined outside the loop, so the pass
// COM: should move the conversion before the loop.
// COM: This hoist *lowers* pressure and so is taken in every GRF mode: the
// COM: #blocked source is 4x fatter per lane than the #dot_op result (256 vs 64
// COM: bytes/lane), and the hoist retires the source from the loop. Projected
// COM: live-in is 288 + 64 - 256 = 96 bytes/lane, matching the measured
// COM: post-hoist figure exactly (loop peak also falls, 484 -> 356).
// COM: This is the substitution accounting from
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7993; an
// COM: additive-only estimate would have projected 352 and rejected at 128-GRF.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_dot_op_cvt
  tt.func @hoist_dot_op_cvt(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<16x16xf16, #dot_b>, %arg2: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %dot : tensor<128x16xf32, #dpas>
    }
    tt.return %result : tensor<128x16xf32, #dpas>
  }
}

// -----

// COM: Case 2: Do NOT hoist ConvertLayoutOp whose destination is NOT DotOperandEncoding.
// COM: The convert_layout goes from #blocked to #dpas (not #dot_op), so it must stay
// COM: inside the loop.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_non_dot_op_cvt
  tt.func @no_hoist_non_dot_op_cvt(%arg0: tensor<128x16xf32, #blocked>, %arg1: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf32, #{{.*}}> -> tensor<128x16xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #dpas>
      %add = arith.addf %cvt, %acc : tensor<128x16xf32, #dpas>
      scf.yield %add : tensor<128x16xf32, #dpas>
    }
    tt.return %result : tensor<128x16xf32, #dpas>
  }
}

// -----

// COM: Case 3: Do NOT hoist ConvertLayoutOp when source is defined inside the loop.

#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 1}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_src_inside_loop
  tt.func @no_hoist_src_inside_loop(%arg0: !tt.ptr<f16>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>>, %arg2: tensor<128x16xf32, #dpas2>) -> tensor<128x16xf32, #dpas2> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: tt.splat
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas2>) : i32 {
      %splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked2>
      %cvt = ttg.convert_layout %splat : tensor<128x16x!tt.ptr<f16>, #blocked2> -> tensor<128x16x!tt.ptr<f16>, #dot_a2>
      scf.yield %acc : tensor<128x16xf32, #dpas2>
    }
    tt.return %result : tensor<128x16xf32, #dpas2>
  }
}

// -----

// COM: Case 4: Do NOT hoist ConvertLayoutOp nested inside scf.if within scf.for.
// COM: The scf.if condition is loop-variant (%iv), so hoisting would make a
// COM: conditional conversion unconditional.

#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a3 = #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>
#dot_b3 = #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_inside_if_variant_cond
  tt.func @no_hoist_inside_if_variant_cond(%arg0: tensor<128x16xf16, #blocked3>, %arg1: tensor<16x16xf16, #dot_b3>, %arg2: tensor<128x16xf32, #dpas3>) -> tensor<128x16xf32, #dpas3> {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: arith.cmpi
    // CHECK: scf.if
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas3>) : i32 {
      %cond = arith.cmpi slt, %iv, %c4_i32 : i32
      %res = scf.if %cond -> (tensor<128x16xf32, #dpas3>) {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked3> -> tensor<128x16xf16, #dot_a3>
        %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a3> * tensor<16x16xf16, #dot_b3> -> tensor<128x16xf32, #dpas3>
        scf.yield %dot : tensor<128x16xf32, #dpas3>
      } else {
        scf.yield %acc : tensor<128x16xf32, #dpas3>
      }
      scf.yield %res : tensor<128x16xf32, #dpas3>
    }
    tt.return %result : tensor<128x16xf32, #dpas3>
  }
}

// -----

// COM: Case 5: Do NOT hoist ConvertLayoutOp nested inside scf.if even with
// COM: loop-invariant condition. The pass conservatively skips any cvt not
// COM: directly in the ForOp's body region.

#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas4 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a4 = #ttg.dot_op<{opIdx = 0, parent = #dpas4, kWidth = 1}>
#dot_b4 = #ttg.dot_op<{opIdx = 1, parent = #dpas4, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_inside_if_invariant_cond
  tt.func @no_hoist_inside_if_invariant_cond(%arg0: tensor<128x16xf16, #blocked4>, %arg1: tensor<16x16xf16, #dot_b4>, %arg2: tensor<128x16xf32, #dpas4>, %cond: i1) -> tensor<128x16xf32, #dpas4> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas4>) : i32 {
      %res = scf.if %cond -> (tensor<128x16xf32, #dpas4>) {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked4> -> tensor<128x16xf16, #dot_a4>
        %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a4> * tensor<16x16xf16, #dot_b4> -> tensor<128x16xf32, #dpas4>
        scf.yield %dot : tensor<128x16xf32, #dpas4>
      } else {
        scf.yield %acc : tensor<128x16xf32, #dpas4>
      }
      scf.yield %res : tensor<128x16xf32, #dpas4>
    }
    tt.return %result : tensor<128x16xf32, #dpas4>
  }
}

// -----

// COM: Case 6: DO hoist at both GRF modes even though the loop body's live-in
// COM: register usage is far past the per-lane GRF budget, because this hoist
// COM: does not spend any of that budget.
// COM: With warpsPerCTA=[1,1] and threadsPerWarp=16, the live-in values to the
// COM: loop body block are (arg2 is NOT live-in — it becomes a block argument
// COM: via iter_args):
// COM:   - arg0 (256x64xf16, blocked):   ~2048 bytes/lane
// COM:   - arg1 (64x16xf16, dot_b):      ~128 bytes/lane
// COM: Live-in total: ~2176 bytes/lane, over budget in every GRF mode (128 GRF:
// COM: 4096/16 * 0.80 = 204 bytes; 256 GRF: 8192/16 * 0.80 = 409 bytes). But the
// COM: hoist retires arg0 and admits an equally wide ~2048-byte #dot_a value, so
// COM: the projection is 2176 + 2048 - 2048 = 2176: exactly break-even. The loop
// COM: is over budget by the same 1767 (resp. 1972) bytes/lane whether or not the
// COM: conversion is hoisted, so the threshold has no say and the conversion is
// COM: hoisted. This is what makes the gate a check on what a hoist *adds*.
// COM:
// COM: The moment where source and result are both live does move out of the
// COM: loop and into the straight-line code before it, and the loop-level gate
// COM: does not measure that program point. The second veto, on the whole
// COM: *function's* peak, does, and here the move lowers that peak: 6272
// COM: bytes/lane before (the loop body, charged for %arg0 through the whole
// COM: loop) and 5248 after (the function block, at the hoisted conversion), as
// COM: no_spanning_in_loop / no_spanning_hoisted in register-pressure.mlir pin.
// COM: The projection: prePeak=6272, corridor=4224 over 2 ops, newPoint=5248,
// COM: ceiling 6272 -> accepted. Case 17 is the same move with a value live
// COM: across the loop.

#blocked6 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas6 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a6 = #ttg.dot_op<{opIdx = 0, parent = #dpas6, kWidth = 1}>
#dot_b6 = #ttg.dot_op<{opIdx = 1, parent = #dpas6, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @over_budget_break_even_hoist
  tt.func @over_budget_break_even_hoist(%arg0: tensor<256x64xf16, #blocked6>, %arg1: tensor<64x16xf16, #dot_b6>, %arg2: tensor<256x16xf32, #dpas6>) -> tensor<256x16xf32, #dpas6> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpas6>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blocked6> -> tensor<256x64xf16, #dot_a6>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_a6> * tensor<64x16xf16, #dot_b6> -> tensor<256x16xf32, #dpas6>
      scf.yield %dot : tensor<256x16xf32, #dpas6>
    }
    tt.return %result : tensor<256x16xf32, #dpas6>
  }
}

// -----

// COM: Case 7: Hoist ConvertLayoutOp when source is a constant defined outside
// COM: the loop. The convert_layout should be moved after the arith.constant.
// COM:
// COM: This also covers a rematerializable source under the substitution model:
// COM: the hoist does retire %cst from the loop, but a constant contributes
// COM: nothing to live-in pressure in the first place (it is filtered out), so
// COM: there is nothing to credit back and the hoist is charged its full cost
// COM: (measured: liveIn=32 + 64 - 0 = 96). Crediting the source's raw type size
// COM: here would have produced a bogus negative.

#blocked7 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas7 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a7 = #ttg.dot_op<{opIdx = 0, parent = #dpas7, kWidth = 1}>
#dot_b7 = #ttg.dot_op<{opIdx = 1, parent = #dpas7, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_constant_source
  tt.func @hoist_constant_source(%arg0: tensor<16x16xf16, #dot_b7>, %arg1: tensor<128x16xf32, #dpas7>) -> tensor<128x16xf32, #dpas7> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked7>
    // CHECK: arith.constant dense<0.000000e+00> : tensor<128x16xf16, #{{.*}}>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<128x16xf32, #dpas7>) : i32 {
      %cvt = ttg.convert_layout %cst : tensor<128x16xf16, #blocked7> -> tensor<128x16xf16, #dot_a7>
      %dot = tt.dot %cvt, %arg0, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a7> * tensor<16x16xf16, #dot_b7> -> tensor<128x16xf32, #dpas7>
      scf.yield %dot : tensor<128x16xf32, #dpas7>
    }
    tt.return %result : tensor<128x16xf32, #dpas7>
  }
}

// -----

// COM: Case 8: Hoist ConvertLayoutOp out of an inner loop when its source is
// COM: loop-invariant w.r.t. the inner loop. The convert_layout moves before
// COM: the inner scf.for but remains inside the outer loop. Hoisting further
// COM: out of the outer loop is left as a follow-up.
// COM: (Same pressure arithmetic as case 1, so likewise taken in every GRF
// COM: mode: the hoist retires the fatter #blocked source from the inner loop.)

#blocked8 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a8 = #ttg.dot_op<{opIdx = 0, parent = #dpas8, kWidth = 1}>
#dot_b8 = #ttg.dot_op<{opIdx = 1, parent = #dpas8, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_from_inner_loop
  tt.func @hoist_from_inner_loop(%arg0: tensor<128x16xf16, #blocked8>, %arg1: tensor<16x16xf16, #dot_b8>, %arg2: tensor<128x16xf32, #dpas8>) -> tensor<128x16xf32, #dpas8> {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %outer = scf.for %oi = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%oacc = %arg2) -> (tensor<128x16xf32, #dpas8>) : i32 {
      %inner = scf.for %ii = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%iacc = %oacc) -> (tensor<128x16xf32, #dpas8>) : i32 {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked8> -> tensor<128x16xf16, #dot_a8>
        %dot = tt.dot %cvt, %arg1, %iacc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a8> * tensor<16x16xf16, #dot_b8> -> tensor<128x16xf32, #dpas8>
        scf.yield %dot : tensor<128x16xf32, #dpas8>
      }
      scf.yield %inner : tensor<128x16xf32, #dpas8>
    }
    tt.return %outer : tensor<128x16xf32, #dpas8>
  }
}

// -----

// COM: Case 9: Do NOT hoist ConvertLayoutOp when the source is an iter_arg.
// COM: Iter args are block arguments of the loop body, so getDefiningOp()
// COM: returns null. They are loop-carried values that change every iteration,
// COM: so the convert_layout must remain inside the loop.

#blocked9 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas9 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a9 = #ttg.dot_op<{opIdx = 0, parent = #dpas9, kWidth = 1}>
#dot_b9 = #ttg.dot_op<{opIdx = 1, parent = #dpas9, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_iter_arg_source
  tt.func @no_hoist_iter_arg_source(%arg0: tensor<128x16xf16, #blocked9>, %arg1: tensor<16x16xf16, #dot_b9>, %arg2: tensor<128x16xf32, #dpas9>) -> tensor<128x16xf32, #dpas9> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter = %arg0) -> (tensor<128x16xf16, #blocked9>) : i32 {
      %cvt = ttg.convert_layout %iter : tensor<128x16xf16, #blocked9> -> tensor<128x16xf16, #dot_a9>
      %dot = tt.dot %cvt, %arg1, %arg2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a9> * tensor<16x16xf16, #dot_b9> -> tensor<128x16xf32, #dpas9>
      %updated = arith.addf %iter, %iter : tensor<128x16xf16, #blocked9>
      scf.yield %updated : tensor<128x16xf16, #blocked9>
    }
    tt.return %arg2 : tensor<128x16xf32, #dpas9>
  }
}

// -----

// COM: Case 10: Hoist ConvertLayoutOp at every GRF mode when the live-ins and
// COM: the hoisted tensor are all small enough to fit comfortably: live-in 96
// COM: bytes/lane, and the hoist swaps a 64-byte source for a 64-byte result, so
// COM: the projection stays at 96 vs. a 204-byte 128-GRF threshold.

#blocked10 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas10 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a10 = #ttg.dot_op<{opIdx = 0, parent = #dpas10, kWidth = 1}>
#dot_b10 = #ttg.dot_op<{opIdx = 1, parent = #dpas10, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_small_tensor
  tt.func @hoist_small_tensor(%arg0: tensor<8x16xf16, #blocked10>, %arg1: tensor<16x16xf16, #dot_b10>, %arg2: tensor<8x16xf32, #dpas10>) -> tensor<8x16xf32, #dpas10> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<8x16xf16, #{{.*}}> -> tensor<8x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<8x16xf32, #dpas10>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<8x16xf16, #blocked10> -> tensor<8x16xf16, #dot_a10>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<8x16xf16, #dot_a10> * tensor<16x16xf16, #dot_b10> -> tensor<8x16xf32, #dpas10>
      scf.yield %dot : tensor<8x16xf32, #dpas10>
    }
    tt.return %result : tensor<8x16xf32, #dpas10>
  }
}

// -----

// COM: Case 11: the divide-by-32 default in getPerLaneGRFBudgetInBytes. The module
// COM: omits ttg.threads-per-warp, so the per-hardware-thread budget is divided by
// COM: 32, not 16. All layouts are 32-lane so the module still verifies.
// COM: %arg0 is also read by the in-loop addf, so the hoist retires nothing: the
// COM: loop live-in is 144 B/lane (%arg0 128 + %arg1 16) and the hoist adds the
// COM: 16 B result, 160.
// COM: The loop gate refuses at 128-GRF (threshold 102 = 80% of 4096 / 32) and
// COM: passes at 256-GRF (threshold 204 = 80% of 8192 / 32). With a divisor of 16
// COM: the 128-GRF threshold is 204 and the hoist would be taken.
// COM: The pre-loop chain %h1..%h4 puts prePeak at 560, above the corridor term
// COM: (448), so the peak gate accepts and the loop gate alone decides.

#blocked11 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas11 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a11 = #ttg.dot_op<{opIdx = 0, parent = #dpas11, kWidth = 1}>
#dot_b11 = #ttg.dot_op<{opIdx = 1, parent = #dpas11, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @default_threads_per_warp
  tt.func @default_threads_per_warp(%arg0: tensor<16x16xf16, #blocked11>, %arg1: tensor<16x16xf16, #dot_b11>, %arg2: tensor<16x16xf32, #dpas11>) -> tensor<16x16xf32, #dpas11> {
    %h1 = arith.addf %arg0, %arg0 : tensor<16x16xf16, #blocked11>
    %h2 = arith.addf %h1, %arg0 : tensor<16x16xf16, #blocked11>
    %h3 = arith.addf %h2, %h1 : tensor<16x16xf16, #blocked11>
    %h4 = arith.addf %h3, %h2 : tensor<16x16xf16, #blocked11>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF128: scf.for
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    %result:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %bacc = %arg0) -> (tensor<16x16xf32, #dpas11>, tensor<16x16xf16, #blocked11>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked11> -> tensor<16x16xf16, #dot_a11>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<16x16xf16, #dot_a11> * tensor<16x16xf16, #dot_b11> -> tensor<16x16xf32, #dpas11>
      %sum = arith.addf %bacc, %arg0 : tensor<16x16xf16, #blocked11>
      scf.yield %dot, %sum : tensor<16x16xf32, #dpas11>, tensor<16x16xf16, #blocked11>
    }
    tt.return %result#0 : tensor<16x16xf32, #dpas11>
  }
}

// -----

// COM: Case 12: Two independent loop-invariant dot operands in one loop, the
// COM: shape of the _attn_bwd_dkdv inner loop (BLOCK_M1=32, BLOCK_N1=64,
// COM: HEAD_DIM=128, num_warps=8). Both conversions are pressure-*neutral*:
// COM: source and result are both 128 bytes/lane, and each hoist retires its own
// COM: source, so each contributes 128 - 128 = 0 to the running per-loop total.
// COM: The second candidate is therefore judged on the same projection as the
// COM: first, and both are hoisted in every GRF mode (measured: loop live-in 256
// COM: bytes/lane before and after, loop peak 1540 -> 1476, function peak
// COM: unchanged at 2560 -- so unlike case 6 this pair costs nothing at the
// COM: hoist site either). At 128-GRF the loop's 256-byte/lane live-in is already
// COM: past the 204-byte threshold, and, as in case 6, that does not veto a pair
// COM: of hoists which leaves it at 256.
// COM:
// COM: This is the regression this accounting fixes. Under additive-only
// COM: accounting the first hoist charged +128 to the loop, which pushed the
// COM: second over the threshold and rejected it, splitting the two operands.
// COM: See https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.
// COM:
// COM: The pass-through iter_args (%q_i, %do_i) are load-bearing, not noise:
// COM: routing %qT and %doT through the loop-carried arguments keeps them out of
// COM: the body's live-in set. Reading them directly would raise live-in from 256
// COM: to 1280 bytes/lane, which is over budget in every GRF mode, and the case
// COM: would then demonstrate nothing.

#blocked12 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#dpas12 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a12 = #ttg.dot_op<{opIdx = 0, parent = #dpas12, kWidth = 1}>
#dot_b12 = #ttg.dot_op<{opIdx = 1, parent = #dpas12, kWidth = 2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @two_neutral_operands
  tt.func @two_neutral_operands(%k: tensor<64x128xf16, #blocked12>, %v: tensor<64x128xf16, #blocked12>,
                                %qT: tensor<128x32xf16, #dot_b12>, %doT: tensor<128x32xf16, #dot_b12>,
                                %dk0: tensor<64x32xf32, #dpas12>, %dv0: tensor<64x32xf32, #dpas12>)
                                -> (tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res:4 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32
        iter_args(%dk = %dk0, %dv = %dv0, %q_i = %qT, %do_i = %doT)
        -> (tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>,
            tensor<128x32xf16, #dot_b12>, tensor<128x32xf16, #dot_b12>) : i32 {
      %kc = ttg.convert_layout %k : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot_a12>
      %qk = tt.dot %kc, %q_i, %dk, inputPrecision = tf32 : tensor<64x128xf16, #dot_a12> * tensor<128x32xf16, #dot_b12> -> tensor<64x32xf32, #dpas12>
      %vc = ttg.convert_layout %v : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot_a12>
      %dp = tt.dot %vc, %do_i, %dv, inputPrecision = tf32 : tensor<64x128xf16, #dot_a12> * tensor<128x32xf16, #dot_b12> -> tensor<64x32xf32, #dpas12>
      scf.yield %qk, %dp, %q_i, %do_i : tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>, tensor<128x32xf16, #dot_b12>, tensor<128x32xf16, #dot_b12>
    }
    tt.return %res#0, %res#1 : tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>
  }
}

// -----

// COM: Case 13: Do NOT credit a retired source that is read *after* the loop.
// COM: Byte-for-byte the same kernel as case 1, plus one use of %arg0 below the
// COM: loop. %arg0 therefore occupies a register for the loop's whole duration
// COM: no matter where the conversion sits, so the hoist frees nothing and earns
// COM: no credit: the projection is the full 288 + 64 = 352 bytes/lane instead of
// COM: case 1's 288 + 64 - 256 = 96, and the hoist is rejected at 128-GRF
// COM: (threshold 204) while still fitting at 256-GRF (threshold 409).
// COM: Measured with -test-register-pressure: the loop body's live-in is 288
// COM: bytes/lane before the hoist and 96 after. That 96 is exactly the figure
// COM: the old accounting projected, and it is the one to distrust: block
// COM: liveness stops counting %arg0 as live-in once no op inside the body reads
// COM: it, even though %arg0's register still has to survive the whole loop to
// COM: feed %post. Real occupancy is 96 + 256 = 352, so the credit undercounted
// COM: it by 3.7x. The measured live-in cannot be used to confirm the projection
// COM: here; the projection is the more faithful number of the two.

#blocked13 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas13 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a13 = #ttg.dot_op<{opIdx = 0, parent = #dpas13, kWidth = 1}>
#dot_b13 = #ttg.dot_op<{opIdx = 1, parent = #dpas13, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_credit_for_source_used_after_loop
  tt.func @no_credit_for_source_used_after_loop(%arg0: tensor<128x16xf16, #blocked13>, %arg1: tensor<16x16xf16, #dot_b13>, %arg2: tensor<128x16xf32, #dpas13>) -> (tensor<128x16xf32, #dpas13>, tensor<128x16xf16, #blocked13>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF256: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF128: scf.for
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas13>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked13> -> tensor<128x16xf16, #dot_a13>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a13> * tensor<16x16xf16, #dot_b13> -> tensor<128x16xf32, #dpas13>
      scf.yield %dot : tensor<128x16xf32, #dpas13>
    }
    // COM: The use that keeps %arg0 live across the loop.
    %post = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked13>
    tt.return %result, %post : tensor<128x16xf32, #dpas13>, tensor<128x16xf16, #blocked13>
  }
}

// -----

// COM: Cases 14 and 15: the same two candidates in one loop, written in the two
// COM: possible orders, must produce the same result. Two candidates:
// COM:   - "lean" %arg0: 128x16xf16 #blocked (256 bytes/lane) -> #dot_a (64), and
// COM:     the hoist retires it, so it contributes 64 - 256 = -192.
// COM:   - "fat" %argF: 16x16xf16 #blocked (32 bytes/lane) -> #dot_b (32), but
// COM:     %argF is also read by an in-loop arith.addf that no hoist can move, so
// COM:     it is never retired and the conversion contributes its full +32.
// COM: Loop live-in is 256 (%arg0) + 32 (%argB) + 32 (%argF) = 320 bytes/lane,
// COM: already past the 128-GRF threshold of 204, so the order of the two
// COM: decisions decides the outcome:
// COM:   - fat first:  320 + 32 = 352 >= 204, rejected -- and a rejection is
// COM:     irrevocable, since it stamps tt.no_licm.
// COM:   - lean first: 320 - 192 = 128 < 204, hoisted; the fat candidate is then
// COM:     judged at 128 + 32 = 160 < 204 and hoisted too.
// COM: The pass therefore decides a loop's candidates cheapest-first rather than
// COM: in program order, and both spellings below hoist both conversions (which
// COM: also fixes the hoisted pair's order: lean before fat, by cost).
// COM: See https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.

#blocked14 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas14 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a14 = #ttg.dot_op<{opIdx = 0, parent = #dpas14, kWidth = 1}>
#dot_b14 = #ttg.dot_op<{opIdx = 1, parent = #dpas14, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Case 14: the fat candidate is written first.
  // CHECK-LABEL: tt.func @order_independent_fat_first
  tt.func @order_independent_fat_first(%arg0: tensor<128x16xf16, #blocked14>, %argB: tensor<16x16xf16, #dot_b14>,
                                       %argF: tensor<16x16xf16, #blocked14>, %arg2: tensor<128x16xf32, #dpas14>)
                                       -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %acc2 = %arg2)
        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) : i32 {
      %fat = ttg.convert_layout %argF : tensor<16x16xf16, #blocked14> -> tensor<16x16xf16, #dot_b14>
      %lean = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked14> -> tensor<128x16xf16, #dot_a14>
      %d1 = tt.dot %lean, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      %d2 = tt.dot %lean, %fat, %acc2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      // COM: Pins %argF inside the loop, so the fat conversion earns no credit.
      %pin = arith.addf %argF, %argF : tensor<16x16xf16, #blocked14>
      scf.yield %d1, %d2 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
    }
    tt.return %res#0, %res#1 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
  }

  // COM: Case 15: the same loop with the lean candidate written first.
  // CHECK-LABEL: tt.func @order_independent_lean_first
  tt.func @order_independent_lean_first(%arg0: tensor<128x16xf16, #blocked14>, %argB: tensor<16x16xf16, #dot_b14>,
                                        %argF: tensor<16x16xf16, #blocked14>, %arg2: tensor<128x16xf32, #dpas14>)
                                        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %acc2 = %arg2)
        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) : i32 {
      %lean = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked14> -> tensor<128x16xf16, #dot_a14>
      %fat = ttg.convert_layout %argF : tensor<16x16xf16, #blocked14> -> tensor<16x16xf16, #dot_b14>
      %d1 = tt.dot %lean, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      %d2 = tt.dot %lean, %fat, %acc2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      // COM: Pins %argF inside the loop, so the fat conversion earns no credit.
      %pin = arith.addf %argF, %argF : tensor<16x16xf16, #blocked14>
      scf.yield %d1, %d2 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
    }
    tt.return %res#0, %res#1 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
  }
}

// -----

// COM: Case 16: one source feeds a conversion in each of two sibling loops, the
// COM: shape of _attn_bwd. Loops are decided last to first, so loop 2 is judged
// COM: first and refused alone: its result would span loop 1, which peaks at 992,
// COM: giving 992 + 64 = 1056 against the ceiling 992. Loop 1 is credited past
// COM: that refused twin and hoisted. The shared-source phase then moves {%cvt2}:
// COM: measured peak 800, accepted. Both conversions land after %src and the
// COM: order differs by mode, so it is not checked.

#blocked16 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas16 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a16 = #ttg.dot_op<{opIdx = 0, parent = #dpas16, kWidth = 1}>
#dot_b16 = #ttg.dot_op<{opIdx = 1, parent = #dpas16, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_source_two_sibling_loops
  tt.func @shared_source_two_sibling_loops(%arg0: tensor<128x16xf16, #blocked16>, %argB: tensor<16x16xf16, #dot_b16>,
                                           %arg2: tensor<128x16xf32, #dpas16>) -> tensor<128x16xf32, #dpas16> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked16>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
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
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas16>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked16> -> tensor<128x16xf16, #dot_a16>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a16> * tensor<16x16xf16, #dot_b16> -> tensor<128x16xf32, #dpas16>
      scf.yield %d2 : tensor<128x16xf32, #dpas16>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas16>
  }
}

// -----


// COM: Case 17: %span (4096 B/lane) lives across the loop without being read
// COM: in it. It is charged through the loop body too, so prePeak (10368) binds
// COM: and the hoist is accepted at the ceiling in every mode (corridor 8320,
// COM: new point 9344). Without that charge prePeak is 8320, under the new
// COM: point, and the hoist is refused. Charging %span twice in the corridor or
// COM: at the new point also crosses 10368. The loop gate is break-even
// COM: (live-in 2176 + 2048 - 2048).

#blocked17 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas17 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a17 = #ttg.dot_op<{opIdx = 0, parent = #dpas17, kWidth = 1}>
#dot_b17 = #ttg.dot_op<{opIdx = 1, parent = #dpas17, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_with_spanning_value
  tt.func @hoist_with_spanning_value(%arg0: tensor<256x64xf16, #blocked17>, %arg1: tensor<64x16xf16, #dot_b17>, %arg2: tensor<256x16xf32, #dpas17>) -> (tensor<256x16xf32, #dpas17>, tensor<256x64xf32, #blocked17>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.extf %arg0 : tensor<256x64xf16, #blocked17> to tensor<256x64xf32, #blocked17>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpas17>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blocked17> -> tensor<256x64xf16, #dot_a17>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_a17> * tensor<64x16xf16, #dot_b17> -> tensor<256x16xf32, #dpas17>
      scf.yield %dot : tensor<256x16xf32, #dpas17>
    }
    tt.return %result, %span : tensor<256x16xf32, #dpas17>, tensor<256x64xf32, #blocked17>
  }
}

// -----


// COM: Case 18: %heavy reads %src between its definition and the loop, so %src
// COM: is live at %heavy wherever the conversion goes and must not be credited
// COM: there. %heavy2 gives %heavy a reader (so it keeps its weight) without
// COM: either value outliving the loop, which would raise prePeak.
// COM: The corridor term at %heavy is 5248 + 2048 = 7296 against prePeak 6272:
// COM: refused. A wrongly granted credit gives 5248 and a hoist.

#blocked18 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas18 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a18 = #ttg.dot_op<{opIdx = 0, parent = #dpas18, kWidth = 1}>
#dot_b18 = #ttg.dot_op<{opIdx = 1, parent = #dpas18, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_credit_while_source_still_read
  tt.func @no_credit_while_source_still_read(%arg0: tensor<256x64xf16, #blocked18>, %arg1: tensor<64x16xf16, #dot_b18>, %arg2: tensor<256x16xf32, #dpas18>) -> tensor<256x16xf32, #dpas18> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blocked18>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %heavy = arith.addf %src, %src : tensor<256x64xf16, #blocked18>
    %heavy2 = arith.addf %heavy, %heavy : tensor<256x64xf16, #blocked18>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpas18>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<256x64xf16, #blocked18> -> tensor<256x64xf16, #dot_a18>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_a18> * tensor<64x16xf16, #dot_b18> -> tensor<256x16xf32, #dpas18>
      scf.yield %dot : tensor<256x16xf32, #dpas18>
    }
    tt.return %result : tensor<256x16xf32, #dpas18>
  }
}

// -----


// COM: Case 19: two independently free candidates in a function whose peak is
// COM: far past the threshold. Both are hoisted.
// COM: This is the in-tree guard for the gate's staleness rule: the analysis is
// COM: rebuilt after each accepted hoist, so the second decision prices its
// COM: corridor and new-point terms on post-move liveness. Reusing the pass-entry
// COM: analysis would price them against liveness that predates the first
// COM: conversion moving. prePeak happens to be 928 either way here, which is why
// COM: what this case pins is the rebuild itself, not a changed ceiling: a stale
// COM: prePeak is unsound because it is a *term* of the projection as well as the
// COM: ceiling, and understating a term understates the maximum.
// COM: Measured: candidate 1 projects 928 (prePeak=928, corridor=608 over 2 ops,
// COM: newPoint=736) and candidate 2 projects 928 (prePeak=928, corridor=736
// COM: over 3 ops, newPoint=736); the ceiling is 928 both times. The loop-level
// COM: gate reports liveIn=288 + thisHoist=-192 = 96 for each.

#blocked19 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas19 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a19 = #ttg.dot_op<{opIdx = 0, parent = #dpas19, kWidth = 1}>
#dot_b19 = #ttg.dot_op<{opIdx = 1, parent = #dpas19, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @two_free_hoists_over_budget
  tt.func @two_free_hoists_over_budget(%arg0: tensor<128x16xf16, #blocked19>, %arg1: tensor<128x16xf16, #blocked19>, %argB: tensor<16x16xf16, #dot_b19>, %acc0: tensor<128x16xf32, #dpas19>) -> tensor<128x16xf32, #dpas19> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %s1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked19>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %s2 = arith.addf %arg1, %arg1 : tensor<128x16xf16, #blocked19>
    // CHECK-NEXT: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas19>) : i32 {
      %cvt1 = ttg.convert_layout %s1 : tensor<128x16xf16, #blocked19> -> tensor<128x16xf16, #dot_a19>
      %d1 = tt.dot %cvt1, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a19> * tensor<16x16xf16, #dot_b19> -> tensor<128x16xf32, #dpas19>
      scf.yield %d1 : tensor<128x16xf32, #dpas19>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %r1) -> (tensor<128x16xf32, #dpas19>) : i32 {
      %cvt2 = ttg.convert_layout %s2 : tensor<128x16xf16, #blocked19> -> tensor<128x16xf16, #dot_a19>
      %d2 = tt.dot %cvt2, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a19> * tensor<16x16xf16, #dot_b19> -> tensor<128x16xf32, #dpas19>
      scf.yield %d2 : tensor<128x16xf32, #dpas19>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas19>
  }
}

// -----


// COM: Case 20: block-argument source with the loop as the block's *first*
// COM: operation, which is the only shape that reaches the projection's
// COM: block-start term. The bounds and the step are block arguments too: a
// COM: single preceding arith.constant would give the hoist a non-null anchor,
// COM: so the new program point would be named by the operation below that
// COM: anchor rather than by the block's first operation.
// COM: Measured: prePeak=556, corridor=364 over 1 op, newPoint=492, ceiling 556.
// COM: The term uses pressureBefore rather than pressureAt precisely because the
// COM: conversion lands *above* the block's first operation: pressureAt takes the
// COM: expansive view that a value is live at its defining operation and would
// COM: have charged the loop's own results there, giving newPoint=620 and
// COM: rejecting a hoist that provably cannot move the reported peak.

#blocked20 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas20 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a20 = #ttg.dot_op<{opIdx = 0, parent = #dpas20, kWidth = 1}>
#dot_b20 = #ttg.dot_op<{opIdx = 1, parent = #dpas20, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_to_block_start
  tt.func @hoist_to_block_start(%src: tensor<128x16xf16, #blocked20>, %argB: tensor<16x16xf16, #dot_b20>, %acc0: tensor<128x16xf32, #dpas20>, %lb: i32, %ub: i32, %st: i32) -> tensor<128x16xf32, #dpas20> {
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %lb to %ub step %st iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas20>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked20> -> tensor<128x16xf16, #dot_a20>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a20> * tensor<16x16xf16, #dot_b20> -> tensor<128x16xf32, #dpas20>
      scf.yield %d : tensor<128x16xf32, #dpas20>
    }
    tt.return %r : tensor<128x16xf32, #dpas20>
  }
}

// -----


// COM: Case 21: an scf.if with a high internal peak sits between %src and the
// COM: loop. The corridor steps over it as one op, so its internal peak is
// COM: priced separately, on top of the point pressure at the if.
// COM: %v reads %src after the if, so %src is not credited there. The if is
// COM: priced at 1568 + 64 = 1632 against prePeak 1568: refused. Without the
// COM: separate pricing it is priced at its own point, 929 + 64, and the hoist
// COM: is accepted.

#blocked21 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas21 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a21 = #ttg.dot_op<{opIdx = 0, parent = #dpas21, kWidth = 1}>
#dot_b21 = #ttg.dot_op<{opIdx = 1, parent = #dpas21, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_over_high_pressure_region
  tt.func @hoist_over_high_pressure_region(%arg0: tensor<128x16xf16, #blocked21>, %argB: tensor<16x16xf16, #dot_b21>, %acc0: tensor<128x16xf32, #dpas21>, %acc1: tensor<128x16xf32, #dpas21>, %cond: i1) -> (tensor<128x16xf32, #dpas21>, tensor<128x16xf32, #dpas21>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked21>
    %hot = scf.if %cond -> (tensor<128x16xf32, #dpas21>) {
      %h1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked21>
      %h2 = arith.addf %h1, %arg0 : tensor<128x16xf16, #blocked21>
      %h3 = arith.addf %h2, %arg0 : tensor<128x16xf16, #blocked21>
      %h4 = arith.addf %h3, %h1 : tensor<128x16xf16, #blocked21>
      %h5 = arith.addf %acc1, %acc1 : tensor<128x16xf32, #dpas21>
      %h6 = ttg.convert_layout %h4 : tensor<128x16xf16, #blocked21> -> tensor<128x16xf16, #dot_a21>
      %h7 = tt.dot %h6, %argB, %h5, inputPrecision = tf32 : tensor<128x16xf16, #dot_a21> * tensor<16x16xf16, #dot_b21> -> tensor<128x16xf32, #dpas21>
      scf.yield %h7 : tensor<128x16xf32, #dpas21>
    } else {
      scf.yield %acc1 : tensor<128x16xf32, #dpas21>
    }
    %v = arith.addf %src, %src : tensor<128x16xf16, #blocked21>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas21>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked21> -> tensor<128x16xf16, #dot_a21>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a21> * tensor<16x16xf16, #dot_b21> -> tensor<128x16xf32, #dpas21>
      scf.yield %d : tensor<128x16xf32, #dpas21>
    }
    tt.return %r, %hot : tensor<128x16xf32, #dpas21>, tensor<128x16xf32, #dpas21>
  }
}

// -----


// COM: Case 22: the source is read *inside* an intervening scf.if, so the
// COM: scf.if itself is the corridor operation that must not be credited.
// COM: findAncestorOpInBlock maps the nested user onto the scf.if, and the
// COM: credit test then asks whether that mapped operation runs strictly before
// COM: the priced one -- which for the scf.if against itself it does not. A bare
// COM: isBeforeInBlock on the nested user would see no ordering at all (the user
// COM: and the scf.if are in different blocks) and would wrongly credit.
// COM: Measured: prePeak=929, corridor=993 over 2 ops, newPoint=737, so the
// COM: projection exceeds the ceiling 929. With the credit wrongly taken the
// COM: corridor term drops to 737 and the hoist is accepted.

#blocked22 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas22 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a22 = #ttg.dot_op<{opIdx = 0, parent = #dpas22, kWidth = 1}>
#dot_b22 = #ttg.dot_op<{opIdx = 1, parent = #dpas22, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_credit_for_src_used_in_intervening_if
  tt.func @no_credit_for_src_used_in_intervening_if(%arg0: tensor<128x16xf16, #blocked22>, %arg1: tensor<128x16xf16, #blocked22>, %argB: tensor<16x16xf16, #dot_b22>, %acc0: tensor<128x16xf32, #dpas22>, %cond: i1) -> (tensor<128x16xf32, #dpas22>, tensor<128x16xf16, #blocked22>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked22>
    %hot = scf.if %cond -> (tensor<128x16xf16, #blocked22>) {
      %u = arith.addf %src, %src : tensor<128x16xf16, #blocked22>
      scf.yield %u : tensor<128x16xf16, #blocked22>
    } else {
      scf.yield %arg1 : tensor<128x16xf16, #blocked22>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas22>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked22> -> tensor<128x16xf16, #dot_a22>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a22> * tensor<16x16xf16, #dot_b22> -> tensor<128x16xf32, #dpas22>
      scf.yield %d : tensor<128x16xf32, #dpas22>
    }
    tt.return %r, %hot : tensor<128x16xf32, #dpas22>, tensor<128x16xf16, #blocked22>
  }
}

// -----


// COM: Case 23: %src is read in the outer loop before the hoist's anchor (%u).
// COM: The outer back edge runs %u again after the inner loop, so the credit
// COM: must be withheld across the whole outer body, not only below %u.
// COM: The corridor term is 608 (544 at the inner loop op plus the 64 B result)
// COM: against prePeak 544: refused. A wrongly granted credit gives 352 and a
// COM: hoist. The dot result is unused so the inner body's peak (480) stays
// COM: below the corridor term instead of mirroring it.

#blocked23 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas23 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a23 = #ttg.dot_op<{opIdx = 0, parent = #dpas23, kWidth = 1}>
#dot_b23 = #ttg.dot_op<{opIdx = 1, parent = #dpas23, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_credit_for_recurring_use_before_anchor
  tt.func @no_credit_for_recurring_use_before_anchor(%src: tensor<128x16xf16, #blocked23>, %argB: tensor<16x16xf16, #dot_b23>, %acc0: tensor<128x16xf32, #dpas23>) -> tensor<128x16xf32, #dpas23> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    %outer = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%o = %acc0) -> (tensor<128x16xf32, #dpas23>) : i32 {
      %u = arith.addf %src, %src : tensor<128x16xf16, #blocked23>
      // CHECK: arith.addf
      // CHECK-NEXT: scf.for
      // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
      %r = scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %o) -> (tensor<128x16xf32, #dpas23>) : i32 {
        %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked23> -> tensor<128x16xf16, #dot_a23>
        %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a23> * tensor<16x16xf16, #dot_b23> -> tensor<128x16xf32, #dpas23>
        scf.yield %a : tensor<128x16xf32, #dpas23>
      }
      scf.yield %r : tensor<128x16xf32, #dpas23>
    }
    tt.return %outer : tensor<128x16xf32, #dpas23>
  }
}

// -----


// COM: Cases 24 and 25: the two branches of the ceiling, which is
// COM: max(prePeak, threshold) and not prePeak alone. Same shape, one extra live
// COM: value apart, both under budget so that the threshold is the binding term.
// COM:
// COM: Case 24 rises but stays within the threshold: measured prePeak=177,
// COM: corridor=193 over 2 ops, newPoint=177, ceiling 204 at 128 GRF. A
// COM: prePeak-only ceiling would reject this hoist even though the function has
// COM: 27 bytes/lane of headroom left after it -- which is the whole reason the
// COM: threshold is part of the ceiling: below it, a rise is not a cost.

#blocked24 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas24 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a24 = #ttg.dot_op<{opIdx = 0, parent = #dpas24, kWidth = 1}>
#dot_b24 = #ttg.dot_op<{opIdx = 1, parent = #dpas24, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @ceiling_is_threshold_accept
  tt.func @ceiling_is_threshold_accept(%arg0: tensor<32x16xf16, #blocked24>, %arg1: tensor<32x16xf16, #blocked24>, %argB: tensor<16x16xf16, #dot_b24>, %acc0: tensor<32x16xf32, #dpas24>, %cond: i1, %e1: tensor<32x16xf16, #blocked24>, %e2: tensor<32x16xf16, #blocked24>, %e3: tensor<32x16xf16, #blocked24>, %e4: tensor<32x16xf16, #blocked24>) -> (tensor<32x16xf32, #dpas24>, tensor<32x16xf16, #blocked24>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked24>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.if
    %hot = scf.if %cond -> (tensor<32x16xf16, #blocked24>) {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked24>
      %u2 = arith.addf %u1, %e2 : tensor<32x16xf16, #blocked24>
      %u3 = arith.addf %u2, %e3 : tensor<32x16xf16, #blocked24>
      %u4 = arith.addf %u3, %e4 : tensor<32x16xf16, #blocked24>
      scf.yield %u4 : tensor<32x16xf16, #blocked24>
    } else {
      scf.yield %arg1 : tensor<32x16xf16, #blocked24>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<32x16xf32, #dpas24>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked24> -> tensor<32x16xf16, #dot_a24>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a24> * tensor<16x16xf16, #dot_b24> -> tensor<32x16xf32, #dpas24>
      scf.yield %d : tensor<32x16xf32, #dpas24>
    }
    tt.return %r, %hot : tensor<32x16xf32, #dpas24>, tensor<32x16xf16, #blocked24>
  }
}

// -----


// COM: Case 25: one more live value in the scf.if than case 24, so the same rise
// COM: crosses the 128-GRF threshold. Measured at 128 GRF: prePeak=193,
// COM: corridor=209 over 2 ops, newPoint=193, ceiling 204 -> rejected. At 256
// COM: GRF the ceiling is 409 and the identical projection is accepted, which is
// COM: what makes this the threshold's own regression test rather than another
// COM: prePeak case.

#blocked25 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas25 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a25 = #ttg.dot_op<{opIdx = 0, parent = #dpas25, kWidth = 1}>
#dot_b25 = #ttg.dot_op<{opIdx = 1, parent = #dpas25, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @ceiling_is_threshold_reject
  tt.func @ceiling_is_threshold_reject(%arg0: tensor<32x16xf16, #blocked25>, %arg1: tensor<32x16xf16, #blocked25>, %argB: tensor<16x16xf16, #dot_b25>, %acc0: tensor<32x16xf32, #dpas25>, %cond: i1, %e1: tensor<32x16xf16, #blocked25>, %e2: tensor<32x16xf16, #blocked25>, %e3: tensor<32x16xf16, #blocked25>, %e4: tensor<32x16xf16, #blocked25>, %e5: tensor<32x16xf16, #blocked25>) -> (tensor<32x16xf32, #dpas25>, tensor<32x16xf16, #blocked25>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked25>
    // GRF256: arith.addf
    // GRF256-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.if
    %hot = scf.if %cond -> (tensor<32x16xf16, #blocked25>) {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked25>
      %u2 = arith.addf %u1, %e2 : tensor<32x16xf16, #blocked25>
      %u3 = arith.addf %u2, %e3 : tensor<32x16xf16, #blocked25>
      %u4 = arith.addf %u3, %e4 : tensor<32x16xf16, #blocked25>
      %u5 = arith.addf %u4, %e5 : tensor<32x16xf16, #blocked25>
      scf.yield %u5 : tensor<32x16xf16, #blocked25>
    } else {
      scf.yield %arg1 : tensor<32x16xf16, #blocked25>
    }
    // CHECK: scf.for
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<32x16xf32, #dpas25>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked25> -> tensor<32x16xf16, #dot_a25>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a25> * tensor<16x16xf16, #dot_b25> -> tensor<32x16xf32, #dpas25>
      scf.yield %d : tensor<32x16xf32, #dpas25>
    }
    tt.return %r, %hot : tensor<32x16xf32, #dpas25>, tensor<32x16xf16, #blocked25>
  }
}

// -----


// COM: Case 26: the source is defined in a *different block* from the loop --
// COM: permitted by the candidate predicate, which only requires the source to be
// COM: loop-invariant. The corridor cannot be walked (its two endpoints are in
// COM: different blocks), so every unwalked point is charged conservatively as
// COM: prePeak + dstBytes and the verdict is a fallback rejection.
// COM: Measured: prePeak=673, corridor=737 over 0 ops (the giveaway that nothing
// COM: was walked), newPoint=481, ceiling 673, "conservatively charged".
// COM: The conservative charge is prePeak + dstBytes rather than prePeak alone
// COM: because the peak point may itself be inside the unwalked corridor.

#blocked26 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas26 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a26 = #ttg.dot_op<{opIdx = 0, parent = #dpas26, kWidth = 1}>
#dot_b26 = #ttg.dot_op<{opIdx = 1, parent = #dpas26, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_hoist_cross_block_source
  tt.func @no_hoist_cross_block_source(%arg0: tensor<128x16xf16, #blocked26>, %argB: tensor<16x16xf16, #dot_b26>, %acc0: tensor<128x16xf32, #dpas26>, %cond: i1) -> tensor<128x16xf32, #dpas26> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked26>
    // CHECK: scf.if
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %res = scf.if %cond -> (tensor<128x16xf32, #dpas26>) {
      %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas26>) : i32 {
        %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked26> -> tensor<128x16xf16, #dot_a26>
        %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a26> * tensor<16x16xf16, #dot_b26> -> tensor<128x16xf32, #dpas26>
        scf.yield %d : tensor<128x16xf32, #dpas26>
      }
      scf.yield %r : tensor<128x16xf32, #dpas26>
    } else {
      scf.yield %acc0 : tensor<128x16xf32, #dpas26>
    }
    tt.return %res : tensor<128x16xf32, #dpas26>
  }
}

// -----


// COM: Case 27: case 26's shape with a zero-byte destination. A pointer-element
// COM: tensor has no int-or-float element type, so getPerThreadSizeInBytes
// COM: prices it at 0 and the conservative charge is prePeak + 0 == prePeak,
// COM: which the ceiling admits. This is the exception that keeps "a fallback
// COM: verdict is a rejection" from being read as a rule: a fallback charges the
// COM: result at every unwalked point, and a result that weighs nothing cannot
// COM: push any of them past the ceiling.
// COM: Measured: prePeak=673, corridor=673 over 0 ops, newPoint=161,
// COM: ceiling 673; the loop-level gate reports liveIn=32 + thisHoist=0 = 32.
// COM: The conversion lands beside its source in the function block, above the
// COM: scf.if that contains the loop.

#blocked27 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas27 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a27 = #ttg.dot_op<{opIdx = 0, parent = #dpas27, kWidth = 1}>
#dot_b27 = #ttg.dot_op<{opIdx = 1, parent = #dpas27, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_zero_byte_destination
  tt.func @hoist_zero_byte_destination(%argptr: tensor<128x16x!tt.ptr<f16>, #blocked27>, %off: tensor<128x16xi32, #blocked27>, %argB: tensor<16x16xf16, #dot_b27>, %acc0: tensor<128x16xf32, #dpas27>, %cond: i1) -> tensor<128x16xf32, #dpas27> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %srcptr = tt.addptr %argptr, %off : tensor<128x16x!tt.ptr<f16>, #blocked27>, tensor<128x16xi32, #blocked27>
    // CHECK: tt.addptr
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16x!tt.ptr<f16>, #{{.*}}> -> tensor<128x16x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.if
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res = scf.if %cond -> (tensor<128x16xf32, #dpas27>) {
      %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas27>) : i32 {
        %cvt = ttg.convert_layout %srcptr : tensor<128x16x!tt.ptr<f16>, #blocked27> -> tensor<128x16x!tt.ptr<f16>, #dot_a27>
        %ld = tt.load %cvt : tensor<128x16x!tt.ptr<f16>, #dot_a27>
        %d = tt.dot %ld, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a27> * tensor<16x16xf16, #dot_b27> -> tensor<128x16xf32, #dpas27>
        scf.yield %d : tensor<128x16xf32, #dpas27>
      }
      scf.yield %r : tensor<128x16xf32, #dpas27>
    } else {
      scf.yield %acc0 : tensor<128x16xf32, #dpas27>
    }
    tt.return %res : tensor<128x16xf32, #dpas27>
  }
}

// -----


// COM: Case 28: the loop sits in a cf.br cycle, where lexical order says nothing
// COM: about what runs again, so %src (read only by the conversion) must not be
// COM: credited as retired.
// COM: The corridor term is 613 (549 at the loop op plus the 64 B result) against
// COM: prePeak 549: refused. A wrongly granted credit gives 357 and a hoist.
// COM: The dot result is unused so the loop body's peak (485) stays below the
// COM: corridor term instead of mirroring it.
// COM: The case where a block's own terminator branches back cannot be written:
// COM: the entry block is not a valid branch target.

#blocked28 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas28 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a28 = #ttg.dot_op<{opIdx = 0, parent = #dpas28, kWidth = 1}>
#dot_b28 = #ttg.dot_op<{opIdx = 1, parent = #dpas28, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_credit_in_unstructured_cfg
  tt.func @no_credit_in_unstructured_cfg(%src: tensor<128x16xf16, #blocked28>, %argB: tensor<16x16xf16, #dot_b28>, %acc0: tensor<128x16xf32, #dpas28>, %n0: i32, %c: i1) -> tensor<128x16xf32, #dpas28> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    cf.br ^header(%acc0, %n0 : tensor<128x16xf32, #dpas28>, i32)
  ^header(%acc: tensor<128x16xf32, #dpas28>, %n: i32):
    cf.cond_br %c, ^body, ^exit
  ^body:
    %u = arith.addi %n, %n : i32
    // CHECK: arith.addi
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc) -> (tensor<128x16xf32, #dpas28>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked28> -> tensor<128x16xf16, #dot_a28>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a28> * tensor<16x16xf16, #dot_b28> -> tensor<128x16xf32, #dpas28>
      scf.yield %a : tensor<128x16xf32, #dpas28>
    }
    cf.br ^header(%r, %u : tensor<128x16xf32, #dpas28>, i32)
  ^exit:
    tt.return %acc : tensor<128x16xf32, #dpas28>
  }
}

// -----


// COM: Cases 29 and 30: the two liveness shapes the projection's result term
// COM: reasons about. The candidate predicate says nothing about how the result
// COM: is used, so both are reachable.
// COM:
// COM: Case 29: a conversion whose result nothing reads. The projection asks the
// COM: analysis what the result weighs instead of deriving it from the type, and
// COM: the analysis charges nothing for a value with no uses -- so dstBytes is 0
// COM: and the hoist cannot move the reported peak at all.
// COM: Measured: prePeak=992, corridor=992 over 2 ops, newPoint=736, ceiling
// COM: 992; the loop-level gate reports liveIn=352 + thisHoist=-192 = 160.
// COM: The peak lives at %heavy, where the source is still read and the corridor
// COM: may not credit it, so charging the result from its type (64 bytes) would
// COM: price that point at 1056 and veto a provably free hoist.

#blocked29 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas29 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a29 = #ttg.dot_op<{opIdx = 0, parent = #dpas29, kWidth = 1}>
#dot_b29 = #ttg.dot_op<{opIdx = 1, parent = #dpas29, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_unused_destination
  tt.func @hoist_unused_destination(%arg0: tensor<128x16xf16, #blocked29>, %extra: tensor<128x16xf16, #blocked29>, %argA: tensor<128x16xf16, #dot_a29>, %argB: tensor<16x16xf16, #dot_b29>, %acc0: tensor<128x16xf32, #dpas29>) -> (tensor<128x16xf32, #dpas29>, tensor<128x16xf16, #blocked29>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked29>
    %heavy = arith.addf %src, %extra : tensor<128x16xf16, #blocked29>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: arith.addf
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas29>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked29> -> tensor<128x16xf16, #dot_a29>
      %d = tt.dot %argA, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a29> * tensor<16x16xf16, #dot_b29> -> tensor<128x16xf32, #dpas29>
      scf.yield %d : tensor<128x16xf32, #dpas29>
    }
    tt.return %r, %heavy : tensor<128x16xf32, #dpas29>, tensor<128x16xf16, #blocked29>
  }
}

// -----


// COM: Case 30: a conversion whose result is *yielded out of the loop*. The
// COM: corridor charges the result at every one of its operations without
// COM: exception, and this is the shape that could make that look wrong: the
// COM: result appears to escape the loop. It does not -- the yield creates a loop
// COM: *result*, a different value -- so the result's live range still begins at
// COM: the conversion and no corridor operation carries it before the move.
// COM: This case is built so that a second charge for the result at the corridor
// COM: operation would show up as a verdict change: source and destination weigh
// COM: the same 16 bytes/lane here (sizePerThread=[1,1] makes the blocked layout
// COM: as narrow as the dot operand), so the credited corridor term lands exactly
// COM: on prePeak. The four %p ballast arguments only lift prePeak past the
// COM: 128-GRF threshold of 204, so that prePeak and not the threshold is the
// COM: ceiling.
// COM: Measured: prePeak=272, corridor=272 over 1 op, newPoint=240, ceiling 272.
// COM: A double charge gives 288 and rejects; so does dropping the credit.

#blocked30 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas30 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a30 = #ttg.dot_op<{opIdx = 0, parent = #dpas30, kWidth = 1}>
#dot_b30 = #ttg.dot_op<{opIdx = 1, parent = #dpas30, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @hoist_yielded_result
  tt.func @hoist_yielded_result(%arg0: tensor<32x16xf16, #blocked30>, %argA: tensor<32x16xf16, #dot_a30>, %argB: tensor<16x16xf16, #dot_b30>, %acc0: tensor<32x16xf32, #dpas30>, %p1: tensor<32x16xf32, #dpas30>, %p2: tensor<32x16xf32, #dpas30>, %p3: tensor<32x16xf32, #dpas30>, %p4: tensor<32x16xf32, #dpas30>) -> (tensor<32x16xf32, #dpas30>, tensor<32x16xf16, #dot_a30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked30>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0, %b = %argA) -> (tensor<32x16xf32, #dpas30>, tensor<32x16xf16, #dot_a30>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked30> -> tensor<32x16xf16, #dot_a30>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a30> * tensor<16x16xf16, #dot_b30> -> tensor<32x16xf32, #dpas30>
      scf.yield %d, %cvt : tensor<32x16xf32, #dpas30>, tensor<32x16xf16, #dot_a30>
    }
    tt.return %r#0, %r#1, %p1, %p2, %p3, %p4 : tensor<32x16xf32, #dpas30>, tensor<32x16xf16, #dot_a30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>, tensor<32x16xf32, #dpas30>
  }
}

// -----


// COM: Case 31: a source with wide fanout. Crediting the departing source means
// COM: scanning its remaining users, and this case pins that the scan is
// COM: unbounded: 17 users still earn the credit. Capping the scan and withholding
// COM: the credit past the cap would charge %hp the arriving result on top of a
// COM: peak it already carries (corridor 992 rather than 736) and reject.
// COM: The %u operations are the fanout. Their results are unused, so the
// COM: analysis charges nothing for them and they cannot perturb the projection
// COM: any other way -- they are there purely to be counted as users.
// COM: %hp is the point the verdict turns on: it is a peak operation that runs
// COM: *after* the last user of %src, which is the only place the credit can
// COM: apply. Per-lane bytes: %src, %e1 and %hp are 256 each, the #dot_a
// COM: result 64, %argB 32, %acc0 and %r 128.
// COM: prePeak is sized to clear the 256-GRF threshold of 409, so the verdict is
// COM: the credit's own rather than the threshold's: a projection below the
// COM: threshold is accepted whatever it costs.
// COM: Measured: prePeak=928, corridor=736 over 19 ops, newPoint=736, ceiling
// COM: 928 -> hoisted.

#blocked31 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas31 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a31 = #ttg.dot_op<{opIdx = 0, parent = #dpas31, kWidth = 1}>
#dot_b31 = #ttg.dot_op<{opIdx = 1, parent = #dpas31, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @credit_with_wide_fanout
  tt.func @credit_with_wide_fanout(%arg0: tensor<128x16xf16, #blocked31>, %e1: tensor<128x16xf16, #blocked31>, %argB: tensor<16x16xf16, #dot_b31>, %acc0: tensor<128x16xf32, #dpas31>) -> (tensor<128x16xf32, #dpas31>, tensor<128x16xf16, #blocked31>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked31>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %u1 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u2 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u3 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u4 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u5 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u6 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u7 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u8 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u9 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u10 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u11 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u12 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u13 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u14 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u15 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u16 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %u17 = arith.addf %src, %src : tensor<128x16xf16, #blocked31>
    %hp = arith.addf %e1, %e1 : tensor<128x16xf16, #blocked31>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas31>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked31> -> tensor<128x16xf16, #dot_a31>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a31> * tensor<16x16xf16, #dot_b31> -> tensor<128x16xf32, #dpas31>
      scf.yield %d : tensor<128x16xf32, #dpas31>
    }
    tt.return %r, %hp : tensor<128x16xf32, #dpas31>, tensor<128x16xf16, #blocked31>
  }
}

// -----


// COM: Case 32: the source is read in the loop body *above* the conversion. The
// COM: hoist earns no retirement credit -- something inside the loop still reads
// COM: the source -- but the withheld credit is only right down to %u1. Below it
// COM: the source is dead, so the corridor operations after %u1 are charged an
// COM: unrelieved source they no longer hold. That is why the rejection lands in
// COM: the fallback bucket rather than being reported as measured.
// COM: Measured at 128 GRF: prePeak=240, corridor=256 over 18 ops, newPoint=144,
// COM: ceiling 240 -> rejected. Piping the 256-GRF output (same projection, wider
// COM: ceiling, so it hoists) through -test-register-pressure gives a post-hoist
// COM: peak of 240, equal to prePeak: the hoist is free and the projection
// COM: overshoots by 16.
// COM: The %w/%s chains exist to lift prePeak past the 204 threshold, so the
// COM: ceiling is prePeak and the verdict turns on the corridor term alone. At
// COM: 256 GRF the ceiling is 409 and the same projection is accepted.

#blocked32 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas32 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a32 = #ttg.dot_op<{opIdx = 0, parent = #dpas32, kWidth = 1}>
#dot_b32 = #ttg.dot_op<{opIdx = 1, parent = #dpas32, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @src_read_in_body_above_cvt
  tt.func @src_read_in_body_above_cvt(%arg0: tensor<32x16xf16, #blocked32>, %argB: tensor<16x16xf16, #dot_b32>, %acc0: tensor<32x16xf32, #dpas32>, %e1: tensor<32x16xf16, #blocked32>, %e2: tensor<32x16xf16, #blocked32>) -> (tensor<32x16xf32, #dpas32>, tensor<32x16xf16, #blocked32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked32>
    // GRF256: arith.addf
    // GRF256-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK: scf.for
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NOT: ttg.convert_layout
    %r:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0, %sp = %arg0) -> (tensor<32x16xf32, #dpas32>, tensor<32x16xf16, #blocked32>) : i32 {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked32>
      %w1 = arith.addf %sp, %e2 : tensor<32x16xf16, #blocked32>
      %w2 = arith.addf %e2, %w1 : tensor<32x16xf16, #blocked32>
      %w3 = arith.addf %e2, %w2 : tensor<32x16xf16, #blocked32>
      %w4 = arith.addf %e2, %w3 : tensor<32x16xf16, #blocked32>
      %w5 = arith.addf %e2, %w4 : tensor<32x16xf16, #blocked32>
      %w6 = arith.addf %e2, %w5 : tensor<32x16xf16, #blocked32>
      %w7 = arith.addf %e2, %w6 : tensor<32x16xf16, #blocked32>
      %w8 = arith.addf %e2, %w7 : tensor<32x16xf16, #blocked32>
      %s1 = arith.addf %w1, %w2 : tensor<32x16xf16, #blocked32>
      %s2 = arith.addf %s1, %w3 : tensor<32x16xf16, #blocked32>
      %s3 = arith.addf %s2, %w4 : tensor<32x16xf16, #blocked32>
      %s4 = arith.addf %s3, %w5 : tensor<32x16xf16, #blocked32>
      %s5 = arith.addf %s4, %w6 : tensor<32x16xf16, #blocked32>
      %s6 = arith.addf %s5, %w7 : tensor<32x16xf16, #blocked32>
      %s7 = arith.addf %s6, %w8 : tensor<32x16xf16, #blocked32>
      %s8 = arith.addf %s7, %u1 : tensor<32x16xf16, #blocked32>
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked32> -> tensor<32x16xf16, #dot_a32>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a32> * tensor<16x16xf16, #dot_b32> -> tensor<32x16xf32, #dpas32>
      scf.yield %d, %s8 : tensor<32x16xf32, #dpas32>, tensor<32x16xf16, #blocked32>
    }
    tt.return %r#0, %r#1 : tensor<32x16xf32, #dpas32>, tensor<32x16xf16, #blocked32>
  }
}

// -----


// COM: Case 33: the source is both the conversion's input and an `iter_args`
// COM: initializer of the loop it is hoisted out of. That read happens before the
// COM: body runs, so once the conversion moves out the source is dead at every
// COM: body point -- the credit is exact, and withholding it because the loop
// COM: appears in the source's user list rejected a free hoist.
// COM: Measured at 128 GRF: prePeak=928, corridor=864 over 3 ops, newPoint=480,
// COM: ceiling 928 -> hoisted. The corridor term now sits *below* prePeak, and
// COM: 864 is exactly the post-hoist peak -test-register-pressure reports.
// COM: Without the credit the corridor term is 992 and the hoist is refused.
// COM: The body is fat on purpose: the two dpas accumulators live across the
// COM: conversion make a body point dominate the corridor, which is where the
// COM: credit applies. Per-lane bytes: %src 256, the #dot_a result 128.

#blocked33 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas33 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a33 = #ttg.dot_op<{opIdx = 0, parent = #dpas33, kWidth = 1}>
#dot_b33 = #ttg.dot_op<{opIdx = 1, parent = #dpas33, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @src_is_also_iter_arg_init
  tt.func @src_is_also_iter_arg_init(%arg0: tensor<128x16xf16, #blocked33>, %argB: tensor<16x16xf16, #dot_b33>, %acc0: tensor<128x16xf32, #dpas33>) -> (tensor<128x16xf32, #dpas33>, tensor<128x16xf16, #blocked33>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked33>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0, %c = %src) -> (tensor<128x16xf32, #dpas33>, tensor<128x16xf16, #blocked33>) : i32 {
      %p1 = arith.addf %a, %a : tensor<128x16xf32, #dpas33>
      %p2 = arith.mulf %a, %p1 : tensor<128x16xf32, #dpas33>
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked33> -> tensor<128x16xf16, #dot_a33>
      %d = tt.dot %cvt, %argB, %p2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a33> * tensor<16x16xf16, #dot_b33> -> tensor<128x16xf32, #dpas33>
      %d2 = arith.addf %d, %p1 : tensor<128x16xf32, #dpas33>
      scf.yield %d2, %c : tensor<128x16xf32, #dpas33>, tensor<128x16xf16, #blocked33>
    }
    tt.return %r#0, %r#1 : tensor<128x16xf32, #dpas33>, tensor<128x16xf16, #blocked33>
  }
}

// -----

// COM: Case 34: %cvt's only use is the first op of an scf.if branch, followed by
// COM: a filler chain in the same branch. The corridor skips the if, so the tail
// COM: after the use is priced on its own, with %src credited: 1441 - 192 = 1249
// COM: against prePeak 1441, accepted. A whole-region charge without the credit
// COM: gives 1441 + 64 = 1505 and refuses.

#blocked34 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas34 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a34 = #ttg.dot_op<{opIdx = 0, parent = #dpas34, kWidth = 1}>
#dot_b34 = #ttg.dot_op<{opIdx = 1, parent = #dpas34, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @price_tail_after_nested_last_use
  tt.func @price_tail_after_nested_last_use(%arg0: tensor<128x16xf16, #blocked34>, %argB: tensor<16x16xf16, #dot_b34>, %acc0: tensor<128x16xf32, #dpas34>, %cond: i1) -> tensor<128x16xf32, #dpas34> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: %[[SRC:.*]] = arith.addf
    // CHECK-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[SRC]] : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked34>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas34>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked34> -> tensor<128x16xf16, #dot_a34>
      // CHECK: scf.if
      // CHECK-NOT: ttg.convert_layout
      // CHECK: tt.dot %[[CVT]]
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas34>) {
        %use = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a34> * tensor<16x16xf16, #dot_b34> -> tensor<128x16xf32, #dpas34>
        %h1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked34>
        %h2 = arith.addf %h1, %arg0 : tensor<128x16xf16, #blocked34>
        %h3 = arith.addf %h2, %arg0 : tensor<128x16xf16, #blocked34>
        %h4 = arith.addf %h3, %h1 : tensor<128x16xf16, #blocked34>
        %h5 = arith.addf %use, %use : tensor<128x16xf32, #dpas34>
        %h6 = ttg.convert_layout %h4 : tensor<128x16xf16, #blocked34> -> tensor<128x16xf16, #dot_a34>
        %h7 = tt.dot %h6, %argB, %h5, inputPrecision = tf32 : tensor<128x16xf16, #dot_a34> * tensor<16x16xf16, #dot_b34> -> tensor<128x16xf32, #dpas34>
        scf.yield %h7 : tensor<128x16xf32, #dpas34>
      } else {
        scf.yield %a : tensor<128x16xf32, #dpas34>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas34>
    }
    tt.return %r : tensor<128x16xf32, #dpas34>
  }
}

// -----


// COM: Case 35: %cvt is used in both branches of an scf.if, so it has no single
// COM: last use and the whole-region fallback is charged: the if's peak plus the
// COM: result, 1441 + 64 = 1505 against prePeak 1441, refused. Dropping the
// COM: result from the fallback gives 1441 and accepts. The filler chain in the
// COM: then-branch lifts the if's peak so prePeak does not mask the charge.

#blocked35 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas35 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a35 = #ttg.dot_op<{opIdx = 0, parent = #dpas35, kWidth = 1}>
#dot_b35 = #ttg.dot_op<{opIdx = 1, parent = #dpas35, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_precise_tail_for_multi_branch_use
  tt.func @no_precise_tail_for_multi_branch_use(%arg0: tensor<128x16xf16, #blocked35>, %argB: tensor<16x16xf16, #dot_b35>, %acc0: tensor<128x16xf32, #dpas35>, %cond: i1) -> tensor<128x16xf32, #dpas35> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked35>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas35>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked35> -> tensor<128x16xf16, #dot_a35>
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas35>) {
        %use1 = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a35> * tensor<16x16xf16, #dot_b35> -> tensor<128x16xf32, #dpas35>
        %h1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked35>
        %h2 = arith.addf %h1, %arg0 : tensor<128x16xf16, #blocked35>
        %h3 = arith.addf %h2, %arg0 : tensor<128x16xf16, #blocked35>
        %h4 = arith.addf %h3, %h1 : tensor<128x16xf16, #blocked35>
        %h5 = arith.addf %use1, %use1 : tensor<128x16xf32, #dpas35>
        %h6 = ttg.convert_layout %h4 : tensor<128x16xf16, #blocked35> -> tensor<128x16xf16, #dot_a35>
        %h7 = tt.dot %h6, %argB, %h5, inputPrecision = tf32 : tensor<128x16xf16, #dot_a35> * tensor<16x16xf16, #dot_b35> -> tensor<128x16xf32, #dpas35>
        scf.yield %h7 : tensor<128x16xf32, #dpas35>
      } else {
        %use2 = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a35> * tensor<16x16xf16, #dot_b35> -> tensor<128x16xf32, #dpas35>
        scf.yield %use2 : tensor<128x16xf32, #dpas35>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas35>
    }
    tt.return %r : tensor<128x16xf32, #dpas35>
  }
}

// -----


// COM: Case 36: %cvt's only use is in the then-branch; the else-branch holds a
// COM: 17-op filler chain that peaks at 865. Once hoisted, %cvt is live through
// COM: the else-branch too: 865 + 64 = 929 against prePeak 865, refused. Without
// COM: pricing the sibling branch the corridor reaches at most 417 and the hoist
// COM: is accepted. %post reads %src, so no source credit applies.

#blocked36 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas36 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a36 = #ttg.dot_op<{opIdx = 0, parent = #dpas36, kWidth = 1}>
#dot_b36 = #ttg.dot_op<{opIdx = 1, parent = #dpas36, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @price_sibling_else_of_nested_last_use
  tt.func @price_sibling_else_of_nested_last_use(%arg0: tensor<128x16xf16, #blocked36>, %argB: tensor<16x16xf16, #dot_b36>, %acc0: tensor<128x16xf32, #dpas36>, %cond: i1) -> (tensor<128x16xf32, #dpas36>, tensor<128x16xf16, #blocked36>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked36>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas36>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked36> -> tensor<128x16xf16, #dot_a36>
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas36>) {
        %use = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a36> * tensor<16x16xf16, #dot_b36> -> tensor<128x16xf32, #dpas36>
        scf.yield %use : tensor<128x16xf32, #dpas36>
      } else {
        %h1 = arith.addf %a, %a : tensor<128x16xf32, #dpas36>
        %h2 = arith.addf %h1, %a : tensor<128x16xf32, #dpas36>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas36>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas36>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas36>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas36>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas36>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas36>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas36>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas36>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas36>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas36>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas36>
        %h14 = arith.addf %h13, %h9 : tensor<128x16xf32, #dpas36>
        %h15 = arith.addf %h14, %h10 : tensor<128x16xf32, #dpas36>
        %h16 = arith.addf %h15, %h11 : tensor<128x16xf32, #dpas36>
        %h17 = arith.addf %h16, %h12 : tensor<128x16xf32, #dpas36>
        scf.yield %h17 : tensor<128x16xf32, #dpas36>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas36>
    }
    %post = arith.addf %src, %src : tensor<128x16xf16, #blocked36>
    tt.return %r, %post : tensor<128x16xf32, #dpas36>, tensor<128x16xf16, #blocked36>
  }
}

// -----


// COM: Case 37: %cvt's only use is at the top of an inner scf.for body, followed
// COM: by a 13-op filler chain. The inner loop's back edge already keeps %cvt
// COM: live over the chain, so the tail must not charge the result again: the
// COM: tail prices at 928 = prePeak, accepted. A double charge gives 928 + 64 =
// COM: 992 and refuses. %post reads %src, so no source credit applies.

#blocked37 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas37 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a37 = #ttg.dot_op<{opIdx = 0, parent = #dpas37, kWidth = 1}>
#dot_b37 = #ttg.dot_op<{opIdx = 1, parent = #dpas37, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_double_charge_for_inner_loop_tail
  tt.func @no_double_charge_for_inner_loop_tail(%arg0: tensor<128x16xf16, #blocked37>, %argB: tensor<16x16xf16, #dot_b37>, %acc0: tensor<128x16xf32, #dpas37>) -> (tensor<128x16xf32, #dpas37>, tensor<128x16xf16, #blocked37>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked37>
    // CHECK: arith.addf
    // CHECK-NEXT: %[[CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas37>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked37> -> tensor<128x16xf16, #dot_a37>
      // CHECK: scf.for
      // CHECK-NOT: ttg.convert_layout
      // CHECK: tt.dot %[[CVT]]
      %hot = scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%b = %a) -> (tensor<128x16xf32, #dpas37>) : i32 {
        %use = tt.dot %cvt, %argB, %b, inputPrecision = tf32 : tensor<128x16xf16, #dot_a37> * tensor<16x16xf16, #dot_b37> -> tensor<128x16xf32, #dpas37>
        %h1 = arith.addf %use, %use : tensor<128x16xf32, #dpas37>
        %h2 = arith.addf %h1, %use : tensor<128x16xf32, #dpas37>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas37>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas37>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas37>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas37>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas37>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas37>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas37>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas37>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas37>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas37>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas37>
        scf.yield %h13 : tensor<128x16xf32, #dpas37>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas37>
    }
    %post = arith.addf %src, %src : tensor<128x16xf16, #blocked37>
    tt.return %r, %post : tensor<128x16xf32, #dpas37>, tensor<128x16xf16, #blocked37>
  }
}

// -----

// COM: Case 38: case 36 mirrored. %cvt's only use is in the else-branch and the
// COM: filler chain is in the then-branch. The sibling is found by block, not by
// COM: branch name: 865 + 64 = 929 against prePeak 865, refused.

#blocked38 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas38 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a38 = #ttg.dot_op<{opIdx = 0, parent = #dpas38, kWidth = 1}>
#dot_b38 = #ttg.dot_op<{opIdx = 1, parent = #dpas38, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @price_sibling_then_of_nested_last_use_in_else
  tt.func @price_sibling_then_of_nested_last_use_in_else(%arg0: tensor<128x16xf16, #blocked38>, %argB: tensor<16x16xf16, #dot_b38>, %acc0: tensor<128x16xf32, #dpas38>, %cond: i1) -> (tensor<128x16xf32, #dpas38>, tensor<128x16xf16, #blocked38>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked38>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas38>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked38> -> tensor<128x16xf16, #dot_a38>
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas38>) {
        %h1 = arith.addf %a, %a : tensor<128x16xf32, #dpas38>
        %h2 = arith.addf %h1, %a : tensor<128x16xf32, #dpas38>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas38>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas38>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas38>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas38>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas38>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas38>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas38>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas38>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas38>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas38>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas38>
        %h14 = arith.addf %h13, %h9 : tensor<128x16xf32, #dpas38>
        %h15 = arith.addf %h14, %h10 : tensor<128x16xf32, #dpas38>
        %h16 = arith.addf %h15, %h11 : tensor<128x16xf32, #dpas38>
        %h17 = arith.addf %h16, %h12 : tensor<128x16xf32, #dpas38>
        scf.yield %h17 : tensor<128x16xf32, #dpas38>
      } else {
        %use = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a38> * tensor<16x16xf16, #dot_b38> -> tensor<128x16xf32, #dpas38>
        scf.yield %use : tensor<128x16xf32, #dpas38>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas38>
    }
    %post = arith.addf %src, %src : tensor<128x16xf16, #blocked38>
    tt.return %r, %post : tensor<128x16xf32, #dpas38>, tensor<128x16xf16, #blocked38>
  }
}

// -----

// COM: Case 39: %cvt's only use is two scf.if levels below %hot, followed by a
// COM: filler chain in %hot's then-branch. The precise tail path requires the use
// COM: to sit one level below %hot, so it must not fire and the whole-region
// COM: fallback is charged: 865 + 64 = 929 against prePeak 865, refused. If the
// COM: precise path fired it would read %cvt as already live on the chain and
// COM: accept at 865.

#blocked39 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas39 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a39 = #ttg.dot_op<{opIdx = 0, parent = #dpas39, kWidth = 1}>
#dot_b39 = #ttg.dot_op<{opIdx = 1, parent = #dpas39, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @depth2_if_if_use_falls_back_soundly
  tt.func @depth2_if_if_use_falls_back_soundly(%arg0: tensor<128x16xf16, #blocked39>, %argB: tensor<16x16xf16, #dot_b39>, %acc0: tensor<128x16xf32, #dpas39>, %cond: i1) -> (tensor<128x16xf32, #dpas39>, tensor<128x16xf16, #blocked39>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked39>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas39>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked39> -> tensor<128x16xf16, #dot_a39>
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas39>) {
        %u = scf.if %cond -> (tensor<128x16xf32, #dpas39>) {
          %use = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a39> * tensor<16x16xf16, #dot_b39> -> tensor<128x16xf32, #dpas39>
          scf.yield %use : tensor<128x16xf32, #dpas39>
        } else {
          scf.yield %a : tensor<128x16xf32, #dpas39>
        }
        %h1 = arith.addf %u, %u : tensor<128x16xf32, #dpas39>
        %h2 = arith.addf %h1, %u : tensor<128x16xf32, #dpas39>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas39>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas39>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas39>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas39>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas39>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas39>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas39>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas39>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas39>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas39>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas39>
        %h14 = arith.addf %h13, %h9 : tensor<128x16xf32, #dpas39>
        %h15 = arith.addf %h14, %h10 : tensor<128x16xf32, #dpas39>
        %h16 = arith.addf %h15, %h11 : tensor<128x16xf32, #dpas39>
        %h17 = arith.addf %h16, %h12 : tensor<128x16xf32, #dpas39>
        scf.yield %h17 : tensor<128x16xf32, #dpas39>
      } else {
        scf.yield %a : tensor<128x16xf32, #dpas39>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas39>
    }
    %post = arith.addf %src, %src : tensor<128x16xf16, #blocked39>
    tt.return %r, %post : tensor<128x16xf32, #dpas39>, tensor<128x16xf16, #blocked39>
  }
}

// -----

// COM: Case 40: two candidates in one loop, each source also read after it.
// COM: %cvtA feeds an scf.if with a filler sibling branch (case 36's shape),
// COM: %cvtB feeds an inner scf.for with a filler chain (case 37's shape).
// COM: At 128-GRF the loop gate refuses both: live-in 161 + 64 = 225 >= 204.
// COM: At 256-GRF %cvtA is refused by the peak (%hotB's region peak 1121 + 64 =
// COM: 1185 against prePeak 1121) and %cvtB is accepted (tail priced at 1121,
// COM: equal to prePeak). Without the already-live check in the tail walk %cvtB
// COM: reads 1185 and is refused too.

#blocked40 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas40 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a40 = #ttg.dot_op<{opIdx = 0, parent = #dpas40, kWidth = 1}>
#dot_b40 = #ttg.dot_op<{opIdx = 1, parent = #dpas40, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @sibling_fix_and_tail_fix_compose
  tt.func @sibling_fix_and_tail_fix_compose(%arg0: tensor<128x16xf16, #blocked40>, %arg1: tensor<128x16xf16, #blocked40>, %argB: tensor<16x16xf16, #dot_b40>, %acc0: tensor<128x16xf32, #dpas40>, %acc1: tensor<128x16xf32, #dpas40>, %cond: i1) -> (tensor<128x16xf32, #dpas40>, tensor<128x16xf32, #dpas40>, tensor<128x16xf16, #blocked40>, tensor<128x16xf16, #blocked40>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: %[[SRCA:.*]] = arith.addf %arg0, %arg0
    %srcA = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked40>
    // CHECK-NEXT: %[[SRCB:.*]] = arith.addf %arg1, %arg1
    %srcB = arith.addf %arg1, %arg1 : tensor<128x16xf16, #blocked40>
    // GRF256: ttg.convert_layout %[[SRCB]] : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF128: scf.for
    %r:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0, %b = %acc1) -> (tensor<128x16xf32, #dpas40>, tensor<128x16xf32, #dpas40>) : i32 {
      // GRF128: ttg.convert_layout %[[SRCA]] {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
      // GRF128-NEXT: ttg.convert_layout %[[SRCB]] {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
      // GRF256: ttg.convert_layout %[[SRCA]] {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
      // GRF256-NEXT: scf.if
      %cvtA = ttg.convert_layout %srcA : tensor<128x16xf16, #blocked40> -> tensor<128x16xf16, #dot_a40>
      %cvtB = ttg.convert_layout %srcB : tensor<128x16xf16, #blocked40> -> tensor<128x16xf16, #dot_a40>
      %hotA = scf.if %cond -> (tensor<128x16xf32, #dpas40>) {
        %h1 = arith.addf %a, %a : tensor<128x16xf32, #dpas40>
        %h2 = arith.addf %h1, %a : tensor<128x16xf32, #dpas40>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas40>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas40>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas40>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas40>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas40>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas40>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas40>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas40>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas40>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas40>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas40>
        %h14 = arith.addf %h13, %h9 : tensor<128x16xf32, #dpas40>
        %h15 = arith.addf %h14, %h10 : tensor<128x16xf32, #dpas40>
        %h16 = arith.addf %h15, %h11 : tensor<128x16xf32, #dpas40>
        %h17 = arith.addf %h16, %h12 : tensor<128x16xf32, #dpas40>
        scf.yield %h17 : tensor<128x16xf32, #dpas40>
      } else {
        %useA = tt.dot %cvtA, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a40> * tensor<16x16xf16, #dot_b40> -> tensor<128x16xf32, #dpas40>
        scf.yield %useA : tensor<128x16xf32, #dpas40>
      }
      %hotB = scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%c = %b) -> (tensor<128x16xf32, #dpas40>) : i32 {
        %useB = tt.dot %cvtB, %argB, %c, inputPrecision = tf32 : tensor<128x16xf16, #dot_a40> * tensor<16x16xf16, #dot_b40> -> tensor<128x16xf32, #dpas40>
        %g1 = arith.addf %useB, %useB : tensor<128x16xf32, #dpas40>
        %g2 = arith.addf %g1, %useB : tensor<128x16xf32, #dpas40>
        %g3 = arith.addf %g2, %g1 : tensor<128x16xf32, #dpas40>
        %g4 = arith.addf %g3, %g2 : tensor<128x16xf32, #dpas40>
        %g5 = arith.addf %g4, %g3 : tensor<128x16xf32, #dpas40>
        %g6 = arith.addf %g5, %g1 : tensor<128x16xf32, #dpas40>
        %g7 = arith.addf %g6, %g2 : tensor<128x16xf32, #dpas40>
        %g8 = arith.addf %g7, %g3 : tensor<128x16xf32, #dpas40>
        %g9 = arith.addf %g8, %g4 : tensor<128x16xf32, #dpas40>
        %g10 = arith.addf %g9, %g5 : tensor<128x16xf32, #dpas40>
        %g11 = arith.addf %g10, %g6 : tensor<128x16xf32, #dpas40>
        %g12 = arith.addf %g11, %g7 : tensor<128x16xf32, #dpas40>
        %g13 = arith.addf %g12, %g8 : tensor<128x16xf32, #dpas40>
        scf.yield %g13 : tensor<128x16xf32, #dpas40>
      }
      scf.yield %hotA, %hotB : tensor<128x16xf32, #dpas40>, tensor<128x16xf32, #dpas40>
    }
    %postA = arith.addf %srcA, %srcA : tensor<128x16xf16, #blocked40>
    %postB = arith.addf %srcB, %srcB : tensor<128x16xf16, #blocked40>
    tt.return %r#0, %r#1, %postA, %postB : tensor<128x16xf32, #dpas40>, tensor<128x16xf32, #dpas40>, tensor<128x16xf16, #blocked40>, tensor<128x16xf16, #blocked40>
  }
}

// -----

// COM: Case 41: case 36 with scf.index_switch. %cvt's only use is in case 0, case
// COM: 1 is light and default holds the 17-op filler chain. Sibling regions are
// COM: priced for any multi-region op: default's 864 + 64 = 928 against prePeak
// COM: 864, refused. %post reads %src, so no source credit applies.

#blocked41 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas41 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a41 = #ttg.dot_op<{opIdx = 0, parent = #dpas41, kWidth = 1}>
#dot_b41 = #ttg.dot_op<{opIdx = 1, parent = #dpas41, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @price_sibling_case_of_index_switch
  tt.func @price_sibling_case_of_index_switch(%arg0: tensor<128x16xf16, #blocked41>, %argB: tensor<16x16xf16, #dot_b41>, %acc0: tensor<128x16xf32, #dpas41>, %sel: index) -> (tensor<128x16xf32, #dpas41>, tensor<128x16xf16, #blocked41>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked41>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas41>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked41> -> tensor<128x16xf16, #dot_a41>
      %hot = scf.index_switch %sel -> tensor<128x16xf32, #dpas41>
      case 0 {
        %use = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a41> * tensor<16x16xf16, #dot_b41> -> tensor<128x16xf32, #dpas41>
        scf.yield %use : tensor<128x16xf32, #dpas41>
      }
      case 1 {
        %k = arith.addf %a, %a : tensor<128x16xf32, #dpas41>
        scf.yield %k : tensor<128x16xf32, #dpas41>
      }
      default {
        %h1 = arith.addf %a, %a : tensor<128x16xf32, #dpas41>
        %h2 = arith.addf %h1, %a : tensor<128x16xf32, #dpas41>
        %h3 = arith.addf %h2, %h1 : tensor<128x16xf32, #dpas41>
        %h4 = arith.addf %h3, %h2 : tensor<128x16xf32, #dpas41>
        %h5 = arith.addf %h4, %h3 : tensor<128x16xf32, #dpas41>
        %h6 = arith.addf %h5, %h1 : tensor<128x16xf32, #dpas41>
        %h7 = arith.addf %h6, %h2 : tensor<128x16xf32, #dpas41>
        %h8 = arith.addf %h7, %h3 : tensor<128x16xf32, #dpas41>
        %h9 = arith.addf %h8, %h4 : tensor<128x16xf32, #dpas41>
        %h10 = arith.addf %h9, %h5 : tensor<128x16xf32, #dpas41>
        %h11 = arith.addf %h10, %h6 : tensor<128x16xf32, #dpas41>
        %h12 = arith.addf %h11, %h7 : tensor<128x16xf32, #dpas41>
        %h13 = arith.addf %h12, %h8 : tensor<128x16xf32, #dpas41>
        %h14 = arith.addf %h13, %h9 : tensor<128x16xf32, #dpas41>
        %h15 = arith.addf %h14, %h10 : tensor<128x16xf32, #dpas41>
        %h16 = arith.addf %h15, %h11 : tensor<128x16xf32, #dpas41>
        %h17 = arith.addf %h16, %h12 : tensor<128x16xf32, #dpas41>
        scf.yield %h17 : tensor<128x16xf32, #dpas41>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas41>
    }
    %post = arith.addf %src, %src : tensor<128x16xf16, #blocked41>
    tt.return %r, %post : tensor<128x16xf32, #dpas41>, tensor<128x16xf16, #blocked41>
  }
}

// -----

// COM: Case 42: %cvt's only use is the init of an inner scf.for, which sits in an
// COM: scf.if. The precise tail pricing starts after that use and would skip the
// COM: inner loop's body, so a last use that holds a region falls back to the
// COM: if's whole-region peak: 1441 + 64 = 1505 against prePeak 1441, refused.
// COM: Without the fallback the corridor term is 673 and the hoist is accepted.

#blocked42 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas42 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a42 = #ttg.dot_op<{opIdx = 0, parent = #dpas42, kWidth = 1}>
#dot_b42 = #ttg.dot_op<{opIdx = 1, parent = #dpas42, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @price_tail_after_nested_last_use_holding_a_region
  tt.func @price_tail_after_nested_last_use_holding_a_region(%arg0: tensor<128x16xf16, #blocked42>, %argB: tensor<16x16xf16, #dot_b42>, %acc0: tensor<128x16xf32, #dpas42>, %cond: i1) -> tensor<128x16xf32, #dpas42> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked42>
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas42>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked42> -> tensor<128x16xf16, #dot_a42>
      %hot = scf.if %cond -> (tensor<128x16xf32, #dpas42>) {
        %inner = scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%c = %cvt) -> (tensor<128x16xf16, #dot_a42>) : i32 {
          %g1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked42>
          %g2 = arith.addf %g1, %arg0 : tensor<128x16xf16, #blocked42>
          %g3 = arith.addf %g2, %arg0 : tensor<128x16xf16, #blocked42>
          %g4 = arith.addf %g3, %g1 : tensor<128x16xf16, #blocked42>
          %g5 = ttg.convert_layout %g4 : tensor<128x16xf16, #blocked42> -> tensor<128x16xf16, #dot_a42>
          scf.yield %g5 : tensor<128x16xf16, #dot_a42>
        }
        %use = tt.dot %inner, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a42> * tensor<16x16xf16, #dot_b42> -> tensor<128x16xf32, #dpas42>
        scf.yield %use : tensor<128x16xf32, #dpas42>
      } else {
        scf.yield %a : tensor<128x16xf32, #dpas42>
      }
      scf.yield %hot : tensor<128x16xf32, #dpas42>
    }
    tt.return %r : tensor<128x16xf32, #dpas42>
  }
}

// -----

// COM: Case 43: case 16 with a third sibling loop. Loops 3 and 2 are each refused
// COM: alone by the peak (992 + 64 = 1056, their result spanning loop 1). Loop 1
// COM: hoists past both refused twins at 992. The shared-source phase then moves
// COM: {%cvt3, %cvt2} together: measured peak 640 + 3*64 + 32 = 864 <= 992. All
// COM: three land after %src at every mode.

#blocked43 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas43 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a43 = #ttg.dot_op<{opIdx = 0, parent = #dpas43, kWidth = 1}>
#dot_b43 = #ttg.dot_op<{opIdx = 1, parent = #dpas43, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_source_three_sibling_loops
  tt.func @shared_source_three_sibling_loops(%arg0: tensor<128x16xf16, #blocked43>, %argB: tensor<16x16xf16, #dot_b43>,
      %arg2: tensor<128x16xf32, #dpas43>) -> tensor<128x16xf32, #dpas43> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked43>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas43>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked43> -> tensor<128x16xf16, #dot_a43>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas43>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas43>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas43>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas43>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas43>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas43>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a43> * tensor<16x16xf16, #dot_b43> -> tensor<128x16xf32, #dpas43>
      scf.yield %d1 : tensor<128x16xf32, #dpas43>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas43>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked43> -> tensor<128x16xf16, #dot_a43>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a43> * tensor<16x16xf16, #dot_b43> -> tensor<128x16xf32, #dpas43>
      scf.yield %d2 : tensor<128x16xf32, #dpas43>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas43>) : i32 {
      %cvt3 = ttg.convert_layout %src : tensor<128x16xf16, #blocked43> -> tensor<128x16xf16, #dot_a43>
      %d3 = tt.dot %cvt3, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a43> * tensor<16x16xf16, #dot_b43> -> tensor<128x16xf32, #dpas43>
      scf.yield %d3 : tensor<128x16xf32, #dpas43>
    }
    tt.return %r3 : tensor<128x16xf32, #dpas43>
  }
}

// -----

// COM: Case 44: case 16 with an unrelated loop between the two siblings. Neither
// COM: the twin credit nor the grouping depends on adjacency, so the outcome is
// COM: case 16's at every mode, ending in {%cvt2} with a measured peak of 800.

#blocked44 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas44 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a44 = #ttg.dot_op<{opIdx = 0, parent = #dpas44, kWidth = 1}>
#dot_b44 = #ttg.dot_op<{opIdx = 1, parent = #dpas44, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_source_non_adjacent_sibling_loops
  tt.func @shared_source_non_adjacent_sibling_loops(%arg0: tensor<128x16xf16, #blocked44>, %argB: tensor<16x16xf16, #dot_b44>,
      %arg2: tensor<128x16xf32, #dpas44>) -> tensor<128x16xf32, #dpas44> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked44>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas44>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked44> -> tensor<128x16xf16, #dot_a44>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas44>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas44>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas44>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas44>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas44>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas44>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a44> * tensor<16x16xf16, #dot_b44> -> tensor<128x16xf32, #dpas44>
      scf.yield %d1 : tensor<128x16xf32, #dpas44>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.addf
    %ru = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas44>) : i32 {
      %u = arith.addf %acc, %acc : tensor<128x16xf32, #dpas44>
      scf.yield %u : tensor<128x16xf32, #dpas44>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %ru) -> (tensor<128x16xf32, #dpas44>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked44> -> tensor<128x16xf16, #dot_a44>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a44> * tensor<16x16xf16, #dot_b44> -> tensor<128x16xf32, #dpas44>
      scf.yield %d2 : tensor<128x16xf32, #dpas44>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas44>
  }
}

// -----

// COM: Case 45: a shared-source group the measured peak refuses. Six sibling
// COM: loops read %src; loop 1 carries case 16's %t chain, loops 2..6 are light.
// COM: Loops 6..2 are each refused alone by the peak (992 + 64 = 1056). Loop 1 is
// COM: credited past those refused twins, which share %src and its result type
// COM: and which remove_layout_conversions merges into %cvt1, and hoists at 992
// COM: at every mode. The shared-source phase then tries {%cvt6 .. %cvt2}:
// COM: measured peak 640 + 6*64 + 32 = 1056 > 992, so the group is moved back and
// COM: all five are stamped. Five siblings would measure 992 and be accepted.
// COM: The MERGE run shows the premise of the credit: the five stamped twins are
// COM: gone and every tt.dot reads %cvt1.

#blocked45 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas45 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a45 = #ttg.dot_op<{opIdx = 0, parent = #dpas45, kWidth = 1}>
#dot_b45 = #ttg.dot_op<{opIdx = 1, parent = #dpas45, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_source_joint_over_peak
  // MERGE-LABEL: tt.func @shared_source_joint_over_peak
  // MERGE: %[[SRC:[0-9]+]] = arith.addf %arg0, %arg0
  // MERGE-NEXT: %[[CVT:[0-9]+]] = ttg.convert_layout %[[SRC]] : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
  // MERGE-NEXT: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE: scf.for
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.dot %[[CVT]],
  // MERGE-NOT: ttg.convert_layout
  // MERGE: tt.return
  tt.func @shared_source_joint_over_peak(%arg0: tensor<128x16xf16, #blocked45>, %argB: tensor<16x16xf16, #dot_b45>,
      %arg2: tensor<128x16xf32, #dpas45>) -> tensor<128x16xf32, #dpas45> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked45>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
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
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d2 : tensor<128x16xf32, #dpas45>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt3 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d3 = tt.dot %cvt3, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d3 : tensor<128x16xf32, #dpas45>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r4 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r3) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt4 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d4 = tt.dot %cvt4, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d4 : tensor<128x16xf32, #dpas45>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r5 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r4) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt5 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d5 = tt.dot %cvt5, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d5 : tensor<128x16xf32, #dpas45>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r6 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r5) -> (tensor<128x16xf32, #dpas45>) : i32 {
      %cvt6 = ttg.convert_layout %src : tensor<128x16xf16, #blocked45> -> tensor<128x16xf16, #dot_a45>
      %d6 = tt.dot %cvt6, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a45> * tensor<16x16xf16, #dot_b45> -> tensor<128x16xf32, #dpas45>
      scf.yield %d6 : tensor<128x16xf32, #dpas45>
    }
    tt.return %r6 : tensor<128x16xf32, #dpas45>
  }
}

// -----

// COM: Case 46: a shared-source group the loop-level gate refuses. Two light
// COM: sibling loops read %src, and so does %keep after both, so no hoist retires
// COM: %src: each loop is 288 + 64 = 352, alone and as a group. The %p chain
// COM: before %src puts the function peak at 928 outside the loops, so the peak
// COM: gate never refuses and the loop gate alone decides.
// COM: At 128-GRF/default 352 >= 204.8: both are refused and the group {%cvt2,
// COM: %cvt1} is moved back unmeasured, all stamped. At 256-GRF 352 < 409.6 and
// COM: both hoist alone.

#blocked46 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas46 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a46 = #ttg.dot_op<{opIdx = 0, parent = #dpas46, kWidth = 1}>
#dot_b46 = #ttg.dot_op<{opIdx = 1, parent = #dpas46, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_source_joint_over_loop_budget
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
    // CHECK: arith.addf %arg0, %arg0
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked46>
    // GRF256-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF256: tt.dot
    // GRF128-NEXT: scf.for
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %p6) -> (tensor<128x16xf32, #dpas46>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked46> -> tensor<128x16xf16, #dot_a46>
      %d1 = tt.dot %cvt1, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a46> * tensor<16x16xf16, #dot_b46> -> tensor<128x16xf32, #dpas46>
      scf.yield %d1 : tensor<128x16xf32, #dpas46>
    }
    // CHECK: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF256: tt.dot
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas46>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked46> -> tensor<128x16xf16, #dot_a46>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a46> * tensor<16x16xf16, #dot_b46> -> tensor<128x16xf32, #dpas46>
      scf.yield %d2 : tensor<128x16xf32, #dpas46>
    }
    // CHECK: arith.addf
    %keep = arith.addf %src, %src : tensor<128x16xf16, #blocked46>
    tt.return %r2 : tensor<128x16xf32, #dpas46>
  }
}

// -----

// COM: Case 47: case 16 with %src a block argument (%arg0). With no defining op a
// COM: hoisted conversion lands right before its own loop, so %cvt2 lands between
// COM: the loops and keeps %arg0 read below loop 1 even when both are moved.
// COM: Loop 2 retires %arg0 (net -192) and hoists at every mode. Loop 1 at
// COM: 128-GRF/default earns no credit, 352 >= 204.8, and is refused; the
// COM: singleton {%cvt1} changes nothing below loop 1, so it is moved back and
// COM: stamped. At 256-GRF 352 < 409.6 and loop 1 hoists.

#blocked47 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas47 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a47 = #ttg.dot_op<{opIdx = 0, parent = #dpas47, kWidth = 1}>
#dot_b47 = #ttg.dot_op<{opIdx = 1, parent = #dpas47, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @shared_block_arg_source_two_sibling_loops
  tt.func @shared_block_arg_source_two_sibling_loops(%arg0: tensor<128x16xf16, #blocked47>, %argB: tensor<16x16xf16, #dot_b47>,
      %arg2: tensor<128x16xf32, #dpas47>) -> tensor<128x16xf32, #dpas47> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF128: scf.for
    // GRF128-NEXT: ttg.convert_layout %arg0 {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256: ttg.convert_layout %arg0 : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas47>) : i32 {
      %cvt1 = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked47> -> tensor<128x16xf16, #dot_a47>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas47>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas47>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas47>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas47>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas47>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas47>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a47> * tensor<16x16xf16, #dot_b47> -> tensor<128x16xf32, #dpas47>
      scf.yield %d1 : tensor<128x16xf32, #dpas47>
    }
    // CHECK: ttg.convert_layout %arg0 : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas47>) : i32 {
      %cvt2 = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked47> -> tensor<128x16xf16, #dot_a47>
      %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a47> * tensor<16x16xf16, #dot_b47> -> tensor<128x16xf32, #dpas47>
      scf.yield %d2 : tensor<128x16xf32, #dpas47>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas47>
  }
}

// -----

// COM: Case 48: the merge charge on a twin's loop. Case 16's loops 1 and 2 on
// COM: %src, plus a second source %s2 (32 B/lane) defined between them and
// COM: converted in loop 2 (%cvtX) and in loop 3 (%cvtZ); %keep reads %s2 after
// COM: every loop, so no hoist retires it. %cvtZ is retyped (repCluster [2, 1])
// COM: so %cvtX is no twin of it.
// COM: At 128-GRF/default %cvt2 is refused by the peak and %cvtX by the loop gate
// COM: (320 + 32 = 352 >= 204). Loop 1 then hoists %cvt1 past the refused twin
// COM: %cvt2 and loop 2 is charged the full -192 the merge is worth. The
// COM: shared-source phase accepts {%cvt2} and then {%cvtX}: 320 - 192 + 32 =
// COM: 160 < 204. A flat +64 charge gives 320 + 64 + 32 = 416 and refuses %cvtX.
// COM: No merge charge at all gives the same outcome as the right one; case 51
// COM: is the shape that needs it. At 256-GRF %cvtX passes the loop gate directly.

#blocked48 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas48 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a48 = #ttg.dot_op<{opIdx = 0, parent = #dpas48, kWidth = 1}>
#dot_b48 = #ttg.dot_op<{opIdx = 1, parent = #dpas48, kWidth = 2}>
#dpas48r = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a48r = #ttg.dot_op<{opIdx = 0, parent = #dpas48r, kWidth = 1}>
#dot_b48r = #ttg.dot_op<{opIdx = 1, parent = #dpas48r, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @twin_merge_charge_frees_sibling_source
  tt.func @twin_merge_charge_frees_sibling_source(%arg0: tensor<128x16xf16, #blocked48>, %argB: tensor<16x16xf16, #dot_b48>,
      %argA: tensor<128x16xf16, #dot_a48r>, %argS2: tensor<16x16xf16, #blocked48>,
      %arg2: tensor<128x16xf32, #dpas48>) -> (tensor<128x16xf32, #dpas48>, tensor<128x16xf32, #dpas48r>, tensor<16x16xf16, #blocked48>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked48>
    // CHECK: arith.addf %arg0, %arg0
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas48>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked48> -> tensor<128x16xf16, #dot_a48>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas48>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas48>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas48>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas48>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas48>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas48>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a48> * tensor<16x16xf16, #dot_b48> -> tensor<128x16xf32, #dpas48>
      scf.yield %d1 : tensor<128x16xf32, #dpas48>
    }
    // CHECK: arith.addf %arg3, %arg3
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %s2 = arith.addf %argS2, %argS2 : tensor<16x16xf16, #blocked48>
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas48>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked48> -> tensor<128x16xf16, #dot_a48>
      %cvtX = ttg.convert_layout %s2 : tensor<16x16xf16, #blocked48> -> tensor<16x16xf16, #dot_b48>
      %d2 = tt.dot %cvt2, %cvtX, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a48> * tensor<16x16xf16, #dot_b48> -> tensor<128x16xf32, #dpas48>
      %e2 = tt.dot %cvt2, %argB, %d2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a48> * tensor<16x16xf16, #dot_b48> -> tensor<128x16xf32, #dpas48>
      scf.yield %e2 : tensor<128x16xf32, #dpas48>
    }
    %zC = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpas48r>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %zC) -> (tensor<128x16xf32, #dpas48r>) : i32 {
      %cvtZ = ttg.convert_layout %s2 : tensor<16x16xf16, #blocked48> -> tensor<16x16xf16, #dot_b48r>
      %d3 = tt.dot %argA, %cvtZ, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a48r> * tensor<16x16xf16, #dot_b48r> -> tensor<128x16xf32, #dpas48r>
      scf.yield %d3 : tensor<128x16xf32, #dpas48r>
    }
    %keep = arith.addf %s2, %s2 : tensor<16x16xf16, #blocked48>
    tt.return %r2, %r3, %keep : tensor<128x16xf32, #dpas48>, tensor<128x16xf32, #dpas48r>, tensor<16x16xf16, #blocked48>
  }
}

// -----

// COM: Case 49: the cascade the twin credit exists for (_attn_bwd's backward
// COM: loop). Case 16's loops on %src, plus %vs (32 B/lane) converted in loop 1
// COM: (%cvtV) and read by %keep after both loops, so %cvtV costs its full +32.
// COM: Loop 2 is refused by the peak. Loop 1 credits %cvt1 past that refused twin
// COM: (-192) and hoists it, which frees loop 1's budget for %cvtV: 320 - 192 +
// COM: 32 = 160 < 204.8. The shared-source phase then moves {%cvt2}.
// COM: Without the credit %cvt1 costs +64, %cvtV is decided first at 320 + 32 =
// COM: 352 >= 204.8 and refused, and %vs is never regrouped: case 50 is that
// COM: counterfactual. At 256-GRF %cvtV clears 352 < 409.6 either way.

#blocked49 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas49 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a49 = #ttg.dot_op<{opIdx = 0, parent = #dpas49, kWidth = 1}>
#dot_b49 = #ttg.dot_op<{opIdx = 1, parent = #dpas49, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @twin_credit_frees_second_candidate
  tt.func @twin_credit_frees_second_candidate(%arg0: tensor<128x16xf16, #blocked49>, %argB: tensor<16x16xf16, #dot_b49>,
      %argV: tensor<16x16xf16, #blocked49>, %arg2: tensor<128x16xf32, #dpas49>)
      -> (tensor<128x16xf32, #dpas49>, tensor<128x16xf32, #dpas49>, tensor<16x16xf16, #blocked49>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked49>
    %vs = arith.addf %argV, %argV : tensor<16x16xf16, #blocked49>
    // CHECK: arith.addf %arg0, %arg0
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: arith.addf %arg2, %arg2
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas49>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked49> -> tensor<128x16xf16, #dot_a49>
      %cvtV = ttg.convert_layout %vs : tensor<16x16xf16, #blocked49> -> tensor<16x16xf16, #dot_b49>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas49>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas49>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas49>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas49>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas49>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas49>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a49> * tensor<16x16xf16, #dot_b49> -> tensor<128x16xf32, #dpas49>
      %e1 = tt.dot %cvt1, %cvtV, %d1, inputPrecision = tf32 : tensor<128x16xf16, #dot_a49> * tensor<16x16xf16, #dot_b49> -> tensor<128x16xf32, #dpas49>
      scf.yield %e1 : tensor<128x16xf32, #dpas49>
    }
    %zB = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #dot_b49>
    %zC = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpas49>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %zC) -> (tensor<128x16xf32, #dpas49>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked49> -> tensor<128x16xf16, #dot_a49>
      %d2 = tt.dot %cvt2, %zB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a49> * tensor<16x16xf16, #dot_b49> -> tensor<128x16xf32, #dpas49>
      scf.yield %d2 : tensor<128x16xf32, #dpas49>
    }
    %keep = arith.addf %vs, %vs : tensor<16x16xf16, #blocked49>
    tt.return %r1, %r2, %keep : tensor<128x16xf32, #dpas49>, tensor<128x16xf32, #dpas49>, tensor<16x16xf16, #blocked49>
  }
}

// -----

// COM: Case 50: no twin credit for the same source converted to a different result
// COM: type. Case 49 with loop 2's %cvt2 retyped to #dot_a50r, whose parent
// COM: differs only in repCluster ([2, 1]); it is still 64 B/lane. The remat
// COM: cache of remove_layout_conversions is keyed on (value, encoding), so it
// COM: never merges %cvt2 into %cvt1 and the credit must not fire.
// COM: At 128-GRF/default %cvt2 counts as a real reader of %src below loop 1, so
// COM: %cvt1 earns no credit (+64): %cvtV is refused at 320 + 32 = 352 >= 204.8,
// COM: then %cvt1 at 320 + 64 = 384. The shared-source phase groups by source
// COM: alone and moves {%cvt2, %cvt1} (measured 864 <= 1056) but never regroups
// COM: %vs, so %cvtV stays stamped; that stamp is what tells this case from case
// COM: 49. The MERGE run keeps both conversions of %src.

#blocked50 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas50 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a50 = #ttg.dot_op<{opIdx = 0, parent = #dpas50, kWidth = 1}>
#dot_b50 = #ttg.dot_op<{opIdx = 1, parent = #dpas50, kWidth = 2}>
#dpas50r = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a50r = #ttg.dot_op<{opIdx = 0, parent = #dpas50r, kWidth = 1}>
#dot_b50r = #ttg.dot_op<{opIdx = 1, parent = #dpas50r, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_twin_credit_for_different_result_type
  // MERGE-LABEL: tt.func @no_twin_credit_for_different_result_type
  // MERGE: %[[SRC:[0-9]+]] = arith.addf %arg0, %arg0
  // MERGE-NEXT: ttg.convert_layout %[[SRC]] : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
  // MERGE-NEXT: ttg.convert_layout %[[SRC]] : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
  tt.func @no_twin_credit_for_different_result_type(%arg0: tensor<128x16xf16, #blocked50>, %argB: tensor<16x16xf16, #dot_b50>,
      %argV: tensor<16x16xf16, #blocked50>, %arg2: tensor<128x16xf32, #dpas50>)
      -> (tensor<128x16xf32, #dpas50>, tensor<128x16xf32, #dpas50r>, tensor<16x16xf16, #blocked50>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked50>
    %vs = arith.addf %argV, %argV : tensor<16x16xf16, #blocked50>
    // CHECK: arith.addf %arg0, %arg0
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: arith.addf %arg2, %arg2
    // GRF256-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas50>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked50> -> tensor<128x16xf16, #dot_a50>
      %cvtV = ttg.convert_layout %vs : tensor<16x16xf16, #blocked50> -> tensor<16x16xf16, #dot_b50>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas50>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas50>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas50>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas50>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas50>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas50>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a50> * tensor<16x16xf16, #dot_b50> -> tensor<128x16xf32, #dpas50>
      %e1 = tt.dot %cvt1, %cvtV, %d1, inputPrecision = tf32 : tensor<128x16xf16, #dot_a50> * tensor<16x16xf16, #dot_b50> -> tensor<128x16xf32, #dpas50>
      scf.yield %e1 : tensor<128x16xf32, #dpas50>
    }
    %zB = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #dot_b50r>
    %zC = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpas50r>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %zC) -> (tensor<128x16xf32, #dpas50r>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked50> -> tensor<128x16xf16, #dot_a50r>
      %d2 = tt.dot %cvt2, %zB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a50r> * tensor<16x16xf16, #dot_b50r> -> tensor<128x16xf32, #dpas50r>
      scf.yield %d2 : tensor<128x16xf32, #dpas50r>
    }
    %keep = arith.addf %vs, %vs : tensor<16x16xf16, #blocked50>
    tt.return %r1, %r2, %keep : tensor<128x16xf32, #dpas50>, tensor<128x16xf32, #dpas50r>, tensor<16x16xf16, #blocked50>
  }
}

// -----

// COM: Case 51: the shape whose outcome needs the merge charge, which case 48's
// COM: does not. Case 48 plus four more light loops (2b..2e) converting %src, so
// COM: the refused twins of %cvt1 are five and their group fails the shared-source
// COM: phase as in case 45. The charge from %cvt1's hoist is then the only thing
// COM: that prices loop 2's merge before {%cvtX} is tried. %cvtZ and loop 3 are
// COM: retyped as in case 48, for the same reason.
// COM: At 128-GRF/default %cvt1 hoists past all five twins, and its hoist charges
// COM: the twins' loops for the merge. The group of five measures 640 + 6*64 +
// COM: 32 + 32 + 64 = 1152 > 1088 and is moved back, stamped. {%cvtX} then reads
// COM: 320 - 192 + 32 = 160 < 204 and hoists. Without the charge it reads
// COM: 320 + 32 = 352 and is stamped; a flat +64 charge reads 416 and stamps it
// COM: too. At 256-GRF %cvtX passes the loop gate directly (352 < 409.6).

#blocked51 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas51 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a51 = #ttg.dot_op<{opIdx = 0, parent = #dpas51, kWidth = 1}>
#dot_b51 = #ttg.dot_op<{opIdx = 1, parent = #dpas51, kWidth = 2}>
#dpas51r = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a51r = #ttg.dot_op<{opIdx = 0, parent = #dpas51r, kWidth = 1}>
#dot_b51r = #ttg.dot_op<{opIdx = 1, parent = #dpas51r, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @merge_charge_outlives_refused_twin_group
  tt.func @merge_charge_outlives_refused_twin_group(%arg0: tensor<128x16xf16, #blocked51>, %argB: tensor<16x16xf16, #dot_b51>,
      %argA: tensor<128x16xf16, #dot_a51r>, %argS2: tensor<16x16xf16, #blocked51>,
      %arg2: tensor<128x16xf32, #dpas51>) -> (tensor<128x16xf32, #dpas51>, tensor<128x16xf32, #dpas51r>, tensor<16x16xf16, #blocked51>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked51>
    // CHECK: arith.addf %arg0, %arg0
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #dpas51>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #dpas51>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas51>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas51>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #dpas51>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas51>
      %d1 = tt.dot %cvt1, %argB, %t6, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %d1 : tensor<128x16xf32, #dpas51>
    }
    // CHECK: arith.addf %arg3, %arg3
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %s2 = arith.addf %argS2, %argS2 : tensor<16x16xf16, #blocked51>
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %cvtX = ttg.convert_layout %s2 : tensor<16x16xf16, #blocked51> -> tensor<16x16xf16, #dot_b51>
      %d2 = tt.dot %cvt2, %cvtX, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      %e2 = tt.dot %cvt2, %argB, %d2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %e2 : tensor<128x16xf32, #dpas51>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2b = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt2b = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %d2b = tt.dot %cvt2b, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %d2b : tensor<128x16xf32, #dpas51>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2c = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2b) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt2c = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %d2c = tt.dot %cvt2c, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %d2c : tensor<128x16xf32, #dpas51>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2d = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2c) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt2d = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %d2d = tt.dot %cvt2d, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %d2d : tensor<128x16xf32, #dpas51>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %r2e = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2d) -> (tensor<128x16xf32, #dpas51>) : i32 {
      %cvt2e = ttg.convert_layout %src : tensor<128x16xf16, #blocked51> -> tensor<128x16xf16, #dot_a51>
      %d2e = tt.dot %cvt2e, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51> * tensor<16x16xf16, #dot_b51> -> tensor<128x16xf32, #dpas51>
      scf.yield %d2e : tensor<128x16xf32, #dpas51>
    }
    %zC = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpas51r>
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %zC) -> (tensor<128x16xf32, #dpas51r>) : i32 {
      %cvtZ = ttg.convert_layout %s2 : tensor<16x16xf16, #blocked51> -> tensor<16x16xf16, #dot_b51r>
      %d3 = tt.dot %argA, %cvtZ, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a51r> * tensor<16x16xf16, #dot_b51r> -> tensor<128x16xf32, #dpas51r>
      scf.yield %d3 : tensor<128x16xf32, #dpas51r>
    }
    %keep = arith.addf %s2, %s2 : tensor<16x16xf16, #blocked51>
    tt.return %r2e, %r3, %keep : tensor<128x16xf32, #dpas51>, tensor<128x16xf32, #dpas51r>, tensor<16x16xf16, #blocked51>
  }
}

// -----

// COM: Case 52: no twin credit for an FMA dot's operands. Case 45's six loops with
// COM: the dot operands' parent a #ttg.blocked layout instead of #dpas, and
// COM: constant B operands. remove_layout_conversions never rematerializes into a
// COM: dot operand whose parent is blocked, so every refused twin stays in its
// COM: loop reading %src: none is excused and each counts as a real reader.
// COM: At 128-GRF/default loops 5..1 earn no credit (256 + 64 = 320 >= 204.8) and
// COM: loop 6 is refused by the peak (960 + 64 = 1024 > 960); the group measures
// COM: 640 + 6*64 = 1024 > 960 and is moved back, all six stamped. A twin credit
// COM: blind to the parent layout would credit loop 1 (256 - 192 = 64) and hoist
// COM: it while its stamped twins keep reading %src inside their loops.

#blocked52 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#fma52 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_a52 = #ttg.dot_op<{opIdx = 0, parent = #fma52}>
#dot_b52 = #ttg.dot_op<{opIdx = 1, parent = #fma52}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_twin_credit_for_blocked_parent_dot_operand
  tt.func @no_twin_credit_for_blocked_parent_dot_operand(%arg0: tensor<128x16xf16, #blocked52>,
      %arg2: tensor<128x16xf32, #fma52>) -> tensor<128x16xf32, #fma52> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %zB = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #dot_b52>
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked52>
    // CHECK: arith.addf %arg0, %arg0
    // GRF256-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    // CHECK-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    // CHECK: tt.dot
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt1 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %t1 = arith.addf %acc, %acc : tensor<128x16xf32, #fma52>
      %t2 = arith.addf %acc, %t1 : tensor<128x16xf32, #fma52>
      %t3 = arith.addf %acc, %t2 : tensor<128x16xf32, #fma52>
      %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #fma52>
      %t5 = arith.addf %acc, %t2 : tensor<128x16xf32, #fma52>
      %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #fma52>
      %d1 = tt.dot %cvt1, %zB, %t6 : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d1 : tensor<128x16xf32, #fma52>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt2 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %d2 = tt.dot %cvt2, %zB, %acc : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d2 : tensor<128x16xf32, #fma52>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r2) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt3 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %d3 = tt.dot %cvt3, %zB, %acc : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d3 : tensor<128x16xf32, #fma52>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    %r4 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r3) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt4 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %d4 = tt.dot %cvt4, %zB, %acc : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d4 : tensor<128x16xf32, #fma52>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    %r5 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r4) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt5 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %d5 = tt.dot %cvt5, %zB, %acc : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d5 : tensor<128x16xf32, #fma52>
    }
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}}>>
    %r6 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r5) -> (tensor<128x16xf32, #fma52>) : i32 {
      %cvt6 = ttg.convert_layout %src : tensor<128x16xf16, #blocked52> -> tensor<128x16xf16, #dot_a52>
      %d6 = tt.dot %cvt6, %zB, %acc : tensor<128x16xf16, #dot_a52> * tensor<16x16xf16, #dot_b52> -> tensor<128x16xf32, #fma52>
      scf.yield %d6 : tensor<128x16xf32, #fma52>
    }
    tt.return %r6 : tensor<128x16xf32, #fma52>
  }
}

// -----

// COM: Case 53: no twin credit past a twin the hoist would not dominate. %arg0 is
// COM: a block argument, so a hoisted conversion lands right above its own loop.
// COM: Loop 1 sits in an scf.if and loop 2 after it, both in an outer loop: %cvt1
// COM: would land inside the scf.if, which does not dominate loop 2, so
// COM: remove_layout_conversions would leave a refused %cvt2 reading %arg0.
// COM: %cvt2 is not excused and counts as a real reader.
// COM: At 128-GRF/default loop 1 therefore earns no credit: 288 + 64 = 352 >=
// COM: 204.8, refused, and the group {%cvt2, %cvt1} is moved back. With the twin
// COM: credit it would read 288 - 192 = 96 and hoist at every mode although
// COM: %arg0 stays read below loop 1. At 256-GRF 352 < 409.6 and loop 1 hoists
// COM: without credit.

#blocked53 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas53 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a53 = #ttg.dot_op<{opIdx = 0, parent = #dpas53, kWidth = 1}>
#dot_b53 = #ttg.dot_op<{opIdx = 1, parent = #dpas53, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_twin_credit_past_a_non_dominated_twin
  tt.func @no_twin_credit_past_a_non_dominated_twin(%arg0: tensor<128x16xf16, #blocked53>, %argB: tensor<16x16xf16, #dot_b53>,
      %acc0: tensor<128x16xf32, #dpas53>, %cond: i1) -> tensor<128x16xf32, #dpas53> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: scf.if
    // GRF256-NEXT: ttg.convert_layout %arg0 : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF128-NEXT: ttg.convert_layout %arg0 {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK: tt.dot
    // CHECK: } else {
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %arg0 {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %out = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%o = %acc0) -> (tensor<128x16xf32, #dpas53>) : i32 {
      %r1 = scf.if %cond -> (tensor<128x16xf32, #dpas53>) {
        %l1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %o) -> (tensor<128x16xf32, #dpas53>) : i32 {
          %cvt1 = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked53> -> tensor<128x16xf16, #dot_a53>
          %d1 = tt.dot %cvt1, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a53> * tensor<16x16xf16, #dot_b53> -> tensor<128x16xf32, #dpas53>
          scf.yield %d1 : tensor<128x16xf32, #dpas53>
        }
        scf.yield %l1 : tensor<128x16xf32, #dpas53>
      } else {
        scf.yield %o : tensor<128x16xf32, #dpas53>
      }
      %l2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas53>) : i32 {
        %cvt2 = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked53> -> tensor<128x16xf16, #dot_a53>
        %d2 = tt.dot %cvt2, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a53> * tensor<16x16xf16, #dot_b53> -> tensor<128x16xf32, #dpas53>
        %t1 = arith.addf %d2, %d2 : tensor<128x16xf32, #dpas53>
        %t2 = arith.addf %d2, %t1 : tensor<128x16xf32, #dpas53>
        %t3 = arith.addf %d2, %t2 : tensor<128x16xf32, #dpas53>
        %t4 = arith.addf %t3, %t1 : tensor<128x16xf32, #dpas53>
        %t5 = arith.addf %d2, %t2 : tensor<128x16xf32, #dpas53>
        %t6 = arith.addf %t4, %t5 : tensor<128x16xf32, #dpas53>
        scf.yield %t6 : tensor<128x16xf32, #dpas53>
      }
      scf.yield %l2 : tensor<128x16xf32, #dpas53>
    }
    tt.return %out : tensor<128x16xf32, #dpas53>
  }
}

// -----

// COM: Case 54: no twin credit past a twin the loop-level gate refused. When a
// COM: conversion's result is per-lane fatter than its source (%s to %t, %h: 8 B
// COM: to 32 B), the merge leaves the twin's loop exactly as hoisting the twin
// COM: would, which is what that gate refused, so the credit must not override it.
// COM: At 128-GRF/default %t is refused by loop 2's gate: 200 + 24 = 224 >= 204.8.
// COM: %h then earns no credit: 176 + 32 = 208 >= 204.8, refused. An
// COM: unconditional twin credit would read 176 + 24 = 200 < 204.8 and hoist %h,
// COM: leaving loop 2 at the 224 its own gate refused.

#blk54 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas54 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a54 = #ttg.dot_op<{opIdx = 0, parent = #dpas54, kWidth = 1}>
#dot_b54 = #ttg.dot_op<{opIdx = 1, parent = #dpas54, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_twin_credit_past_a_loop_gate_refusal
  tt.func @no_twin_credit_past_a_loop_gate_refusal(%arg0: tensor<16x16xf16, #blk54>, %w0: tensor<16x16xf16, #blk54>,
      %a1: tensor<128x16xf16, #dot_a54>, %a2: tensor<128x16xf16, #dot_a54>, %a3: tensor<128x16xf16, #dot_a54>,
      %argB: tensor<16x16xf16, #dot_b54>, %acc0: tensor<128x16xf32, #dpas54>) -> (tensor<128x16xf32, #dpas54>, tensor<16x16xf16, #blk54>) {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: arith.addf %arg0, %arg0
    // GRF256-NEXT: ttg.convert_layout %{{[0-9a-z_]+}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: arith.addf %arg1, %arg1
    // CHECK-NEXT: scf.for
    // GRF256-NOT: ttg.convert_layout
    // GRF128-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK: tt.dot
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %s = arith.addf %arg0, %arg0 : tensor<16x16xf16, #blk54>
    %w = arith.addf %w0, %w0 : tensor<16x16xf16, #blk54>
    %r1 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc0) -> (tensor<128x16xf32, #dpas54>) : i32 {
      %h = ttg.convert_layout %s : tensor<16x16xf16, #blk54> -> tensor<16x16xf16, #dot_b54>
      %d = tt.dot %a1, %h, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a54> * tensor<16x16xf16, #dot_b54> -> tensor<128x16xf32, #dpas54>
      %e = tt.dot %a2, %argB, %d, inputPrecision = tf32 : tensor<128x16xf16, #dot_a54> * tensor<16x16xf16, #dot_b54> -> tensor<128x16xf32, #dpas54>
      %wx = arith.addf %w, %w : tensor<16x16xf16, #blk54>
      scf.yield %e : tensor<128x16xf32, #dpas54>
    }
    %r2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas54>) : i32 {
      %t = ttg.convert_layout %s : tensor<16x16xf16, #blk54> -> tensor<16x16xf16, #dot_b54>
      %d = tt.dot %a1, %t, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a54> * tensor<16x16xf16, #dot_b54> -> tensor<128x16xf32, #dpas54>
      %e = tt.dot %a2, %t, %d, inputPrecision = tf32 : tensor<128x16xf16, #dot_a54> * tensor<16x16xf16, #dot_b54> -> tensor<128x16xf32, #dpas54>
      %f = tt.dot %a3, %t, %e, inputPrecision = tf32 : tensor<128x16xf16, #dot_a54> * tensor<16x16xf16, #dot_b54> -> tensor<128x16xf32, #dpas54>
      scf.yield %f : tensor<128x16xf32, #dpas54>
    }
    tt.return %r2, %w : tensor<128x16xf32, #dpas54>, tensor<16x16xf16, #blk54>
  }
}
