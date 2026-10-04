// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" | FileCheck %s --check-prefixes=CHECK,GRF128
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128" | FileCheck %s --check-prefixes=CHECK,GRF128
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=256" | FileCheck %s --check-prefixes=CHECK,GRF256
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" -tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=MERGE

// COM: The MERGE run chains this pass with remove_layout_conversions, the
// COM: later pass whose backward-rematerialization cache the credit in
// COM: hoistRetiresSource relies on to replace a refused "twin" (same source,
// COM: same result type) with the hoisted conversion that dominates it. Only
// COM: cases 45 and 50 carry MERGE checks.

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
// STATS: [HoistLayoutConversions] considered=95 hoisted=47 rejected_pressure=16 rejected_function_peak_exact=17 rejected_function_peak_fallback=6 skipped_other=9
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
// COM: does not measure that program point. What the move costs is measured
// COM: separately, by the second veto on the whole *function's* peak -- and here
// COM: it costs nothing. Measured whole-function peak: 5248 bytes/lane both
// COM: before and after the hoist. Only the block reporting it changes; the loop
// COM: body block drops 5248 -> 4224 while the function block rises 4224 -> 5248.
// COM: The veto's projection reproduces that: prePeak=5248, corridor=4224 over
// COM: 1 op, newPoint=5248, ceiling 5248 -> accepted.
// COM: Case 17 is the shape in which the same move does raise the function peak,
// COM: and is refused for it.

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

// COM: Case 11: Exercise the divide-by-32 default in
// COM: RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes. This module
// COM: deliberately omits ttg.threads-per-warp, so getThreadsPerWarp returns its
// COM: 32 default and the per-hardware-thread budget is divided by 32 rather than
// COM: the DPAS-typical 16. All layouts here are 32-lane so the module still
// COM: verifies (deleting the attribute from a SIMD16 module would not).
// COM:
// COM: Measured pressure: live-in 144 bytes/lane (arg0 #blocked 128 + arg1
// COM: #dot_b 16; the iter args are block arguments, so they are not live-in),
// COM: plus 16 bytes/lane for the hoisted #dot_a value = 160 bytes/lane total.
// COM:
// COM: %arg0 is deliberately also consumed by the in-loop arith.addf, so the
// COM: hoist does NOT retire it and the cost is genuinely additive (this is also
// COM: the coverage for the not-retired branch of the substitution model). Were
// COM: the convert_layout its only in-loop use, the hoist would instead swap a
// COM: 128-byte source for a 16-byte result and pass at every GRF mode.
// COM:
// COM: Thresholds are 80% of the per-lane budget:
// COM:   grf-mode=default -> 4096/32 = 128 B/lane -> threshold 102
// COM:   grf-mode=256     -> 8192/32 = 256 B/lane -> threshold 204
// COM: Were the divisor wrongly 16, the thresholds would be 204 and 409; the
// COM: GRF128 rejection below (160 exceeds 80% of 128) is unaffected by the
// COM: region-aware fix and still pins the divide-by-32 behavior on its own.
// COM:
// COM: GRF256 no longer hoists, though: this same %arg0, read directly inside
// COM: the loop by the addf above (the same "deliberately also consumed"
// COM: reference), is now correctly charged live through the loop's entire
// COM: duration, which raises prePeak from 416 to 432 and reveals that the
// COM: corridor (448, unaffected by the loop-level threshold math above) does
// COM: exceed even that higher ceiling. So the whole-function peak veto now
// COM: rejects this hoist at every GRF mode, for a reason unrelated to the
// COM: divide-by-32 arithmetic this case exists to pin -- that arithmetic is
// COM: only demonstrated by the GRF128 side of the original contrast now, not
// COM: by a GRF128-rejects/GRF256-hoists split.

#blocked11 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas11 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a11 = #ttg.dot_op<{opIdx = 0, parent = #dpas11, kWidth = 1}>
#dot_b11 = #ttg.dot_op<{opIdx = 1, parent = #dpas11, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @default_threads_per_warp
  tt.func @default_threads_per_warp(%arg0: tensor<16x16xf16, #blocked11>, %arg1: tensor<16x16xf16, #dot_b11>, %arg2: tensor<16x16xf32, #dpas11>) -> tensor<16x16xf32, #dpas11> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
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

// COM: Case 16: One loop-invariant source feeding a conversion in each of two
// COM: *sibling* loops. This is the shape _attn_bwd has at HEAD_DIM=128, where
// COM: every hoisting source (k, v, q, do) is read by an adjacent pair of loops.
// COM:
// COM: The credit for retiring %src is real for both loops, but only if the
// COM: question "is %src still read below this loop?" is asked of the IR as it
// COM: stands rather than of the frozen liveness analysis. For the *earlier* loop
// COM: the honest answer depends on the later loop's conversion having already
// COM: moved out from under it, so the pass decides loops last-to-first: loop 2 is
// COM: considered first (nothing below it reads %src), then loop 1. See
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7993 for the
// COM: last-to-first ordering this case was originally added to pin down.
// COM:
// COM: Per-lane bytes: %src is 256, each #dot_a result is 64, each loop body's
// COM: live-in is 256 (%src) + 32 (%argB) = 288 (the accumulators are iter args,
// COM: hence block arguments, hence not live-in). Uncredited, a hoist's own
// COM: loop-level budget check compares 288 + 64 = 352 against 80% of the
// COM: per-lane GRF budget, which only 256-GRF's 512 B/lane clears (409.6);
// COM: 128-GRF and default's 256 B/lane (204.8) reject it. Credited, it is
// COM: 288 + 64 - 256 = 96, a delta of -192 that the gate never vetoes. Loop
// COM: 2's hoist is always credited (nothing below it reads %src), so only the
// COM: whole-function peak veto below can refuse it, and does, at every GRF
// COM: mode.
// COM:
// COM: %src is produced by an arith.addf, giving it a defining operation, so
// COM: %cvt1's hoisted landing point is right after that definition rather
// COM: than just above loop 1 itself. Either position is before loop 2 here,
// COM: so this case does not exercise the block-argument dominance
// COM: restriction (a block-argument source's hoisted conversion only
// COM: dominates a twin inside its own loop or after it in the same block,
// COM: landing just above that loop instead of below a defining op) -- see
// COM: case 53 for that restriction actually biting.
// COM:
// COM: Both hoists must also clear the whole-function peak veto, and the %t
// COM: chain in loop 1's body is what makes that a real test rather than a
// COM: formality. Five 128-byte #dpas values are live at once at %t4 (640
// COM: B/lane); %src is *also* live through the whole of loop 1, a direct read
// COM: rather than merely forwarded, so it is charged for the loop's entire
// COM: duration regardless of where %cvt1 sits (+256 B/lane). Together that is
// COM: the function's peak, 992 B/lane -- 256 higher than this case's pre-fix
// COM: number (736) because the old analysis under-counted %src as not live at
// COM: %t4 at all.
// COM:
// COM: Loop 2's hoist moves %cvt2 above loop 1, to just after %src's
// COM: definition, so once hoisted alone %cvt2's own result is *also* live
// COM: through the whole of loop 1 -- unused there, but occupying a register
// COM: for its entire duration, the same back-edge rule that charges %src. That
// COM: is a genuinely new 64 B/lane charge at %t4, pushing the projected peak
// COM: to 1056 and vetoing loop 2's hoist, decided alone, at every GRF mode:
// COM: prePeak=992, corridor=1056 over 3 ops (loop 1, stepped over as a sibling
// COM: region and priced by its own internal peak), newPoint=480; ceiling 992.
// COM: Loop 1's hoist, decided alone, clears the veto exactly at the ceiling:
// COM: projected 992 (prePeak=992, corridor=608 over 2 ops, newPoint=480). Its
// COM: loop-level gate does not refuse it either. Loop 2's refused %cvt2 has
// COM: %cvt1's exact (source, result type), so remove_layout_conversions
// COM: merges %cvt2 into the hoisted %cvt1 later on, and hoistRetiresSource
// COM: excuses that unenforceable twin as a reader of %src. %cvt1 is
// COM: credited, 288 + 64 - 256 = 96, and hoisted at every GRF mode. The merge
// COM: charge then prices loop 2 at the same -192 the merge is worth there and
// COM: marks %cvt2 merge-charged.
// COM:
// COM: Deciding one loop at a time therefore strands %cvt2 at every mode, yet
// COM: hoisting it as well genuinely lowers the function's peak: once neither
// COM: loop reads %src, its 256 bytes retire for real and two 64-byte results
// COM: replace it, 640 + 2*64 + 32 (%argB) = 800 B/lane at %t4, against 992
// COM: before. (Issue #8202; before the shared-source phase the pass stopped
// COM: at 992 at every mode, and only a second run of the pass reached 800, and
// COM: only at 256-GRF.) %src already has a hoisted conversion, so the
// COM: shared-source phase tries the singleton group {%cvt2}. Its loop is
// COM: merge-charged, so the loop-level gate skips it, and the measured peak
// COM: is 800 <= ceiling 992. Accepted, at every mode.
// COM: So both conversions land after %src at every mode, neither stamped.
// COM: The order of the two hoisted conversions differs by mode (each move lands
// COM: directly after %src), so the checks do not pin it.

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


// COM: Case 17: originally the motivating case for the whole-function peak
// COM: veto, and the shape gap (b) of
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7993
// COM: describes: a fat value that spans the loop without being read inside it.
// COM: %span is not the conversion's source; its job is to be live *around*
// COM: the loop.
// COM:
// COM: With the region-aware register pressure fix, this case no longer
// COM: demonstrates a veto: %span spans the loop entirely unused, so it is now
// COM: correctly charged as live through the loop's *own* body too (the same
// COM: back-edge accounting `RegisterPressureAnalysis::getLiveThroughAncestorSet`
// COM: applies to any value referenced -- or, as here, merely spanning -- a
// COM: loop), which raises prePeak from the old, under-counted 6272 to 8320 (the
// COM: +2048 is exactly %span). The new program point's figure is unchanged at
// COM: newPoint=7296 -- still lower than the now-correct prePeak -- so the
// COM: projection (max of prePeak=8320, corridor=6272 over 2 ops, newPoint=7296)
// COM: is exactly 8320, at the ceiling, and the hoist is accepted in every GRF
// COM: mode: the loop already carried this much pressure with or without the
// COM: hoist, which the old analysis simply could not see.
// COM: Per-lane bytes (warpsPerCTA=[1,1]): %arg0 and the #dot_a result are both
// COM: 2048, %span 2048, %arg1 128, %arg2 128.
// COM: The loop-level gate has no say either way: this hoist is break-even on
// COM: the loop body's live-in pressure (2176 + 2048 - 2048), exactly case 6's
// COM: arithmetic, so `hoistDeltaBytes` is 0 and the budget check at
// COM: HoistLayoutConversions.cpp's `bestDelta > 0` guard never triggers.

#blocked17 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas17 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a17 = #ttg.dot_op<{opIdx = 0, parent = #dpas17, kWidth = 1}>
#dot_b17 = #ttg.dot_op<{opIdx = 1, parent = #dpas17, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_hoist_spanning_value
  tt.func @no_hoist_spanning_value(%arg0: tensor<256x64xf16, #blocked17>, %arg1: tensor<64x16xf16, #dot_b17>, %arg2: tensor<256x16xf32, #dpas17>) -> (tensor<256x16xf32, #dpas17>, tensor<256x64xf16, #blocked17>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blocked17>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpas17>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blocked17> -> tensor<256x64xf16, #dot_a17>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_a17> * tensor<64x16xf16, #dot_b17> -> tensor<256x16xf32, #dpas17>
      scf.yield %dot : tensor<256x16xf32, #dpas17>
    }
    tt.return %result, %span : tensor<256x16xf32, #dpas17>, tensor<256x64xf16, #blocked17>
  }
}

// -----


// COM: Case 18: unsound-credit regression. %heavy reads the conversion's source
// COM: between the source's definition and the loop, so at %heavy the source is
// COM: still live no matter where the conversion goes and the corridor may not
// COM: credit it there.
// COM:
// COM: An earlier version of this case returned %heavy after the loop, making
// COM: it *also* a value spanning the loop entirely unused -- exactly case 17's
// COM: shape -- so the region-aware fix correctly charged it as live through
// COM: the loop's own body too, raising prePeak from 6272 to 8320 (exactly
// COM: %heavy's own 2048 bytes). Being prePeak, that floor cannot be exceeded
// COM: by any hoist, so the decision flipped from reject to accept regardless
// COM: of whether the credit-withholding mechanism below still worked: with
// COM: it correctly withholding credit, corridor was 7296, still comfortably
// COM: under the now-8320 ceiling; had it been wrongly credited instead
// COM: (7296 - 2048 = 5248), the verdict would have been identical. The test
// COM: could no longer tell the two apart.
// COM:
// COM: %heavy2 fixes this: it gives %heavy a real second reader (so %heavy
// COM: keeps its own weight at its own corridor entry, rather than being
// COM: filtered to 0 bytes as an unused value) without either value surviving
// COM: past the loop, so neither is case 17's shape and prePeak stays at the
// COM: lower, un-inflated 6272.
// COM: Measured: prePeak=6272, corridor=7296 over 4 ops, newPoint=5248; ceiling
// COM: 6272 -- corridor now dominates and exceeds it, so the hoist is refused,
// COM: with credit still correctly withheld at %heavy's own entry. Had it been
// COM: wrongly credited there instead using %src's own real size (2048 bytes,
// COM: the same type as %heavy's, per its own entry above): 7296 - 2048 = 5248,
// COM: well under the 6272 ceiling instead of exceeding it, so the verdict
// COM: would flip to accept -- the two answers land on opposite sides of the
// COM: threshold again.

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
    // %heavy must still read %src between %src's definition and the loop (the
    // whole point of this case), and must still carry its own real weight at
    // that point (consumed by %heavy2, also before the loop) so that entry
    // stays the dominant one -- but neither may survive to be read *after*
    // the loop, or it becomes case 17's shape (a value spanning the loop
    // unused) and inflates prePeak enough to mask the credit-withholding
    // check this case exists to pin.
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


// COM: Case 21: an intervening scf.if with high local pressure that reads
// COM: neither the source nor the result. The corridor steps over the region as
// COM: one operation and does not descend into it.
// COM: That rule is load-bearing here: pressure lives *inside* the if body, at
// COM: %h3, where four 256-byte values are live at once. Before the
// COM: sibling-region-pricing fix, the per-op corridor walk's point-pressure
// COM: query did not report that peak at all (the hoisted result spans the
// COM: region unused), so the whole-function peak this candidate was checked
// COM: against read as 1184, and adding %h3's own internal peak on top would
// COM: have priced it at 1184 + 64 = 1248. With the fix, `projectedFunctionPeak`
// COM: does separately check %hot's own internal peak against this candidate
// COM: via `regionPeakThroughOp` -- see the real, current figures a few
// COM: paragraphs below (1568, not 1184; corridor 1376, not 1248) -- so the
// COM: "don't fail to price a stepped-over sibling's own high pressure" point this
// COM: case was built to pin is still true and still demonstrated.
// COM:
// COM: %hot itself is a value that spans the *second* loop (%r) entirely
// COM: unused -- it is defined before %r and not read again until the final
// COM: tt.return, after %r closes -- exactly case 17/18's shape, just with the
// COM: spanned region being a different loop than the one being hoisted from.
// COM: With the region-aware fix, %hot is correctly charged as live through
// COM: %r's whole body, which raises prePeak to 1568 -- a bound this candidate
// COM: cannot lower, since %hot's live range lives in %r's body untouched by
// COM: this hoist.
// COM:
// COM: %src, read only by %cvt (this candidate) and nothing else, is *also*
// COM: live through the if -- not because the if reads it, but because %cvt,
// COM: still sitting inside %r at this point in the walk, reads it later in
// COM: the same block. That is credited away: the same rule the flat per-op
// COM: charge above uses (nothing but %cvt itself still needs %src past the
// COM: if, so the credit is exact, not conservative) applies equally to a
// COM: stepped-over sibling's own internal peak, dropping the if's own
// COM: internal peak by %src's 256 bytes and adding the hoisted result's 64:
// COM: regionPeakThroughOp(if) + 64 - 256 = 1568 - 256 + 64 = 1376.
// COM: Measured: prePeak=1568, corridor=1376 over 3 ops, newPoint=865, ceiling
// COM: 1568 -- prePeak now dominates, and this break-even hoist sits exactly at
// COM: the ceiling, so it is accepted in every GRF mode, matching main.

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
    // CHECK: arith.addf
    // CHECK-NEXT: %[[CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.if
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
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot %[[CVT]]
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


// COM: Case 23: the source is read inside an intervening loop that *encloses*
// COM: the corridor, at a point lexically before the hoist's anchor. Position
// COM: says nothing here -- %u runs again on the next iteration, after the inner
// COM: loop -- so the credit is withheld for every operation of the enclosing
// COM: loop's body rather than only for those below the last user. The corridor
// COM: and new-point terms below are exactly this credit-withholding mechanism,
// COM: and they are unchanged by the region-aware fix (corridor is still 1376,
// COM: newPoint still 1248): the source is still never wrongly credited.
// COM:
// COM: What did change is prePeak, from 1312 to 1440. %src is read directly
// COM: inside the *outer* loop (%u, %w, %x), so it is now correctly charged as
// COM: live through the outer loop's entire body -- including transitively
// COM: inside the nested inner loop, which is exactly the multi-level nesting
// COM: gap #8053 exists to fix. That +128 (%src's own per-lane contribution)
// COM: makes prePeak the dominant term and, being prePeak, an unbeatable floor:
// COM: the function already carries this much pressure regardless of the hoist,
// COM: so the hoist is now accepted (at the ceiling exactly) in every GRF mode,
// COM: to a point *inside* the outer loop, immediately above the inner loop.
// COM: Measured: prePeak=1440, corridor=1376 over 2 ops, newPoint=1248, ceiling
// COM: 1440.

#blocked23 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas23 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a23 = #ttg.dot_op<{opIdx = 0, parent = #dpas23, kWidth = 1}>
#dot_b23 = #ttg.dot_op<{opIdx = 1, parent = #dpas23, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func @no_credit_for_recurring_use_before_anchor
  tt.func @no_credit_for_recurring_use_before_anchor(%src: tensor<128x16xf16, #blocked23>, %arg1: tensor<128x16xf16, #blocked23>, %argB: tensor<16x16xf16, #dot_b23>, %acc0: tensor<128x16xf32, #dpas23>) -> (tensor<128x16xf32, #dpas23>, tensor<128x16xf16, #blocked23>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    %outer:2 = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%o = %acc0, %p = %arg1) -> (tensor<128x16xf32, #dpas23>, tensor<128x16xf16, #blocked23>) : i32 {
      %u = arith.addf %src, %p : tensor<128x16xf16, #blocked23>
      %w = arith.addf %src, %u : tensor<128x16xf16, #blocked23>
      %x = arith.addf %src, %w : tensor<128x16xf16, #blocked23>
      // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
      // CHECK-NEXT: scf.for
      // CHECK-NOT: ttg.convert_layout
      %r = scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %o) -> (tensor<128x16xf32, #dpas23>) : i32 {
        %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked23> -> tensor<128x16xf16, #dot_a23>
        %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a23> * tensor<16x16xf16, #dot_b23> -> tensor<128x16xf32, #dpas23>
        scf.yield %d : tensor<128x16xf32, #dpas23>
      }
      %y = arith.addf %u, %w : tensor<128x16xf16, #blocked23>
      %z = arith.addf %y, %x : tensor<128x16xf16, #blocked23>
      scf.yield %r, %z : tensor<128x16xf32, #dpas23>, tensor<128x16xf16, #blocked23>
    }
    tt.return %outer#0, %outer#1 : tensor<128x16xf32, #dpas23>, tensor<128x16xf16, #blocked23>
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


// COM: Case 28: an unstructured CFG cycle on the region path above the corridor.
// COM: The credit test bails out on any enclosing region that is not a single
// COM: block whose terminator has no CFG successors, because inside a cycle
// COM: lexical position carries no information about what runs again.
// COM: ^body has exactly one predecessor and the back edge leaves from it, so a
// COM: multi-predecessor test on the corridor's own block would miss this. %src
// COM: has no user other than the conversion, so without the bail-out the credit
// COM: would be granted unconditionally at every corridor operation.
// COM: Measured: prePeak=613, corridor=613 over 2 ops, newPoint=485, ceiling
// COM: 613. With the credit taken the corridor term is 357.
// COM: The credit-withholding bail-out this case exists to pin is unaffected by
// COM: the region-aware fix: the corridor term is exactly what it always was
// COM: (613, unchanged), since it is `pressureAt` queried at the scf.for's own
// COM: program point in ^bb2, which was already correct -- MLIR's own raw
// COM: liveness already treats a value read inside a nested region as live at
// COM: the region-holding op's point, independent of any ancestor-chain
// COM: tracking. So the credit is still correctly withheld and the hoist is
// COM: still gated on the same, unimproved corridor figure.
// COM:
// COM: What now separately pushes the decision to accept is prePeak, which
// COM: rose from the old, under-counted 549 to 613: %src is read directly
// COM: inside the scf.for's body, so -- like case 1's own loop, which this
// COM: case otherwise mirrors -- it is now correctly charged for the loop's
// COM: entire duration, including at the point past its own last local use
// COM: where the loop's own dot result is produced (`peakPressure` charges a
// COM: value at its defining op, so the dot's own 128 B/lane result is added
// COM: on top). Being prePeak, that is an unbeatable floor once it dominates:
// COM: the loop already carries this much pressure regardless of the hoist.
// COM: The %u = arith.addi anchor keeps the pre-loop operations *lean* on
// COM: purpose. Inside a CFG cycle every value used anywhere in the cycle is
// COM: reported live at every operation of it, including values read only on a
// COM: later iteration, so a fat operation before the loop would outrank the loop
// COM: itself and move prePeak away from the point this case is about.
// COM:
// COM: The degenerate companion -- a single block whose *own* terminator has CFG
// COM: successors, which pins the second half of the same bail-out -- has no test
// COM: because it cannot be written: the entry block of a region is not a valid
// COM: branch target, so `cf.br ^bb0` in a one-block tt.func is rejected by the
// COM: parser with "reference to an undefined block".

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
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc) -> (tensor<128x16xf32, #dpas28>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<128x16xf16, #blocked28> -> tensor<128x16xf16, #dot_a28>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a28> * tensor<16x16xf16, #dot_b28> -> tensor<128x16xf32, #dpas28>
      scf.yield %d : tensor<128x16xf32, #dpas28>
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

// COM: Case 34 (#8053 follow-up): %cvt's only use is early inside the
// COM: scf.if's then-branch, and a chain of high-pressure fillers (%h1..%h4)
// COM: follows it, later in the *same* branch, before the if closes.
// COM: lastBodyUser maps that nested use onto the scf.if itself, so
// COM: collectCorridor excludes the if op -- and everything from %cvt up to
// COM: it -- from the corridor, on the theory that the un-hoisted result is
// COM: already locally live there at the same cost hoisting would add. That
// COM: theory only holds up to the real nested use; the filler chain runs
// COM: strictly after it, still inside the same branch, and needs the same
// COM: post-last-use pricing any other corridor entry gets. Before either fix
// COM: that interval was priced as zero, since the if op was never a corridor
// COM: entry at all (neither the per-op walk nor the region-peak fallback
// COM: -- which only fires for corridor entries -- ever reaches it).
// COM:
// COM: %cvt's real (not top-level-mapped) last use is %use, at the very top
// COM: of the branch; realLastUse finds it directly since it is %cvt's only
// COM: use, and priceTailAfter then prices only the filler chain after it
// COM: (%h1..%h7, the yield), crediting %src the same exact way the flat
// COM: corridor charge does -- %cvt is %src's only reader, so nothing needs
// COM: %src past the if either.
// COM: Measured at default GRF: prePeak=1441, corridor=1249 over 2 ops,
// COM: newPoint=737 -- prePeak now dominates and this break-even hoist sits
// COM: exactly at the ceiling, so it is accepted, matching main.
// COM: A whole-region charge with no credit and no tail restriction (the
// COM: fallback this case exercised before the tail fix) would instead price
// COM: the if's full internal peak plus %cvt's own bytes twice over (once
// COM: already counted from %use up to %h5, once added back on top): 1505,
// COM: which exceeds the ceiling and wrongly refuses a hoist main accepts.

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


// COM: Case 35 (#8053 follow-up): %cvt has two uses in *different* branches of
// COM: the same scf.if (one in "then", one in "else"), with the same
// COM: high-pressure filler chain (%h1..%h4) as case 34 in the "then" branch
// COM: to make sure this case's own charge, not just prePeak, decides it (see
// COM: below). realLastUse requires every use to share one immediate block to
// COM: place a single last-use position; here the two users' blocks differ,
// COM: so it returns null and priceTailAfter's precise path is skipped in
// COM: favor of the conservative, uncredited whole-region charge -- the
// COM: fallback case 34's fix deliberately declines to make precise.
// COM: An earlier version of this case had no filler chain and measured
// COM: prePeak=673, corridor=673: an exact tie that only pinned "the fallback
// COM: fires and does not crash," not its arithmetic -- dropping `+ dstBytes`
// COM: entirely from the fallback's charge, an undercount, left corridor at
// COM: 609, still masked by the tied prePeak, and the test kept passing.
// COM: With the filler chain raising the if's own internal peak, that
// COM: masking is gone: dropping `+ dstBytes` the same way now drops corridor
// COM: to 1441, exactly prePeak, and wrongly accepts -- caught.
// COM: Measured at default GRF: prePeak=1441, corridor=1505 over 2 ops,
// COM: newPoint=737 -- corridor dominates and exceeds the 1441 ceiling, so
// COM: the hoist is refused. %src (256 bytes) is read in both branches, so
// COM: main cannot retire it either; refusing here is not a missed
// COM: optimization, just a conservative charge landing on the safe side.

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


// COM: Case 36 (#8053 follow-up, independent-review F1): `%cvt`'s only use is
// COM: in the "then" branch of `%hot`, `realLastUse` succeeds, and
// COM: `priceTailAfter` prices the (empty) tail after it precisely -- but
// COM: never looks at the "else" branch at all, since `%hot` is never itself a
// COM: corridor entry once its real nested last use is found. Once hoisted,
// COM: `%cvt` is loop-invariant and, by the loop back-edge rule, live through
// COM: the *whole* loop body, "else" included, where a 17-op filler chain
// COM: (unrelated to `%cvt`/`%src`) drives that branch's own peak up. Before
// COM: the fix this silently undercounted by exactly `dstBytes` (64) and
// COM: tripped the `FunctionPeakGate` invariant assert in a Debug build
// COM: (`freshPeak <= pendingProjection`); a Release build would have
// COM: accepted a hoist that overruns the ceiling by 64 B/lane.
// COM: `%src` is also read after the loop (in `%post`) so no source credit
// COM: applies, matching the shape that actually reaches this code path in
// COM: practice (credit alone cannot mask the gap; see the independent
// COM: review's own note that crediting `%src` away hides the bug).
// COM: Measured at every GRF mode: prePeak=865, corridor=929 over 2 ops
// COM: (the "then" tail, empty, and the "else" sibling peak, 865 + 64),
// COM: newPoint=289; ceiling 865. corridor now dominates and matches the
// COM: real post-hoist peak (929, measured by hand-hoisting through
// COM: `-test-register-pressure`) exactly, so the hoist is correctly refused.

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


// COM: Case 37 (#8053 follow-up, independent-review F2): `%cvt`'s only use is
// COM: at the top of an *inner* `scf.for`'s body (`%hot`), followed by a
// COM: 13-op filler chain reading only the dot's own result, not `%cvt`
// COM: again. `realLastUse` succeeds and `priceTailAfter` walks the filler
// COM: chain -- but the inner loop is itself a loop ancestor, so the
// COM: region-aware analysis this PR adds already keeps `%cvt` live at every
// COM: op of that chain *before* any hoisting, via the same loop back-edge
// COM: rule that makes the hoisted result live through the outer loop.
// COM: Charging `dstBytes` unconditionally at each op of the chain therefore
// COM: double-counts a weight the pre-hoist measurement already includes.
// COM: Before the fix this refused a hoist that costs nothing (measured by
// COM: hand-hoisting: the peak is unchanged at 928) and reported the refusal
// COM: as `rejected_function_peak_exact`, i.e. as if it were certain rather
// COM: than an artifact of the double charge.
// COM: `%src` is also read after the loop so no source credit applies,
// COM: isolating the double-charge from the (separately tested) credit path.
// COM: Measured at every GRF mode: prePeak=928, corridor=928 over 2 ops,
// COM: newPoint=288; ceiling 928. The hoist is now accepted, matching the
// COM: unchanged real peak.

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

// COM: Case 38 (#8053 follow-up, broader F1 coverage): the mirror image of
// COM: case 36 -- `%cvt`'s only use now sits in the *else* branch of `%hot`,
// COM: and the 17-op filler chain (unrelated to `%cvt`/`%src`) sits in "then"
// COM: instead. `siblingBlocksPeak` keys off block identity (the block being
// COM: excluded is `nestedLast`'s own block), not a "then"/"else" name, so
// COM: this checks the fix is not accidentally tied to which branch happens
// COM: to hold the real use. `%src` is also read after the loop (in `%post`)
// COM: so no source credit applies, same as case 36.
// COM: Measured at every GRF mode: prePeak=865, corridor=929 over 2 ops (the
// COM: "else" tail, empty, and the "then" sibling peak, 865 + 64), newPoint=
// COM: 289; ceiling 865. Verified by hand-hoisting through
// COM: -test-register-pressure: real pre-hoist peak 865, real post-hoist peak
// COM: 929, exact match to the corridor term. The hoist is correctly refused.

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

// COM: Case 39 (#8053 follow-up, broader F2/depth coverage; replaces an
// COM: earlier version of this case per an independent verification review,
// COM: `review_opus55_verification_2026-09-27.md` V4): `%cvt`'s only use is
// COM: nested two `scf.if` levels below `%hot`, the op `lastBodyUser` maps
// COM: it to -- an inner `scf.if` sits inside `%hot`'s "then", and the real
// COM: use is inside that inner `scf.if`'s own "then". A 17-op filler chain,
// COM: seeded from the inner `scf.if`'s result, follows it in `%hot`'s
// COM: "then", *after* the inner `scf.if` closes but still at depth one
// COM: relative to `%hot`.
// COM: The precise path's own precondition (`nestedLast`'s block's parent
// COM: op must equal `lastUse`) correctly fails here -- the real use's
// COM: block's parent is the inner `scf.if`, not `%hot` -- so this exercises
// COM: the fallback (logged as "conservatively charged", `exact=false`).
// COM: Measured at every GRF mode: prePeak=865, corridor=929 over 2 ops,
// COM: newPoint=289; ceiling 865, so the hoist is refused. Hand-hoisting
// COM: through -test-register-pressure confirms real pre-hoist peak 865 and
// COM: real post-hoist peak 929 -- the fallback is exact here, not just
// COM: safe, and matches the true cost of hoisting through two `scf.if`
// COM: levels: the outer filler chain runs *after* the real last use, at a
// COM: point where the un-hoisted value has already died (it is not itself
// COM: a loop, so there is no back-edge keeping it live), so it does need
// COM: the full `dstBytes` charge once hoisted.
// COM: This case exists to show the depth precondition is load-bearing for
// COM: *soundness*, not just a CHECK-line change detector: with the
// COM: precondition removed (`nestedLast->getBlock()->getParentOp() ==
// COM: lastUse` deleted), the precise path wrongly fires using
// COM: `siblingBlocksPeak(lastUse=%hot, excluded=innerIfBlock)`, which reads
// COM: `rep = block.front()` of `%hot`'s "then" block to ask whether `%cvt`
// COM: is already live there -- and since `%cvt` *is* still live at that
// COM: point (it is read later, inside the inner `scf.if`), the check
// COM: answers "already live" and skips `dstBytes` on the filler chain that
// COM: actually runs after `%cvt`'s last use. That wrongly accepts at 865
// COM: (the real peak is 929) and trips the Debug `FunctionPeakGate`
// COM: invariant assert (`freshPeak <= pendingProjection`); verified
// COM: directly by temporarily deleting the precondition and rebuilding.
// COM: The previous version of this case (two `scf.for` levels, `scf.if`
// COM: inside `scf.for` inside `scf.for`) did not have this property: with
// COM: the same precondition removed, it still only fell back to a *safe*
// COM: (992) rather than exact answer, because an enclosing loop keeps
// COM: `%cvt` live everywhere in its body regardless of depth, masking
// COM: whether the precondition itself was doing any work. `scf.if` does
// COM: not have that back-edge rule, which is why nesting through `scf.if`
// COM: levels instead is what actually discriminates the precondition.

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

// COM: Case 40 (#8053 follow-up, F1+F2 composition): two independent hoist
// COM: candidates in the same loop. `%cvtA` is shaped like case 36/38 (feeds
// COM: an `scf.if`, whose sibling branch needs `siblingBlocksPeak`'s
// COM: pricing); `%cvtB` is shaped like case 37 (feeds an inner `scf.for`,
// COM: whose tail walk needs `priceTailAfter`'s already-live check to avoid
// COM: double-charging). `lastUse` cannot be both an `scf.if` and an
// COM: `scf.for` for the *same* candidate -- it is a single, specific op --
// COM: so this is the meaningful way the two fixes compose: on two different
// COM: candidates in one pass invocation, sharing the same per-generation
// COM: `QueryCache`/`regionPeakCache`, checking neither fix corrupts the
// COM: other's cache entries or fights over the shared candidate-ordering
// COM: and rebuild-after-accept bookkeeping.
// COM: At GRF128 (default and 128): both candidates are rejected before
// COM: reaching either mechanism -- their combined loop-carried live-in alone
// COM: (161 B/lane) plus either one's 64-byte destination exceeds 80% of the
// COM: 256 B/lane budget -- so both stay in place (counted as
// COM: rejected_pressure, not skipped_other; confirmed via
// COM: TRITON_INTEL_HLC_STATS=1).
// COM: At GRF256: `%cvtA` is considered first (program order) and correctly
// COM: rejected (prePeak=1121, corridor=1185 over 4 ops, exact -- not
// COM: fallback); `%cvtB` is then considered against the same, unchanged
// COM: analysis (nothing was hoisted yet) and correctly accepted via the
// COM: already-live check in `priceTailAfter` (prePeak=corridor=1121, i.e.
// COM: the hoist is free). Verified by running the pass's actual transformed
// COM: output through -test-register-pressure: real peak is 1121 both before
// COM: and after the pass, matching the "free hoist" verdict for %cvtB, and
// COM: by inspecting the IR dump directly: %cvtA (fed from %arg0) keeps its
// COM: `tt.no_licm` marker inside the loop, while %cvtB (fed from %arg1) is
// COM: the one hoisted above `scf.for`.
// COM: Checked which mechanism is actually load-bearing for each verdict by
// COM: disabling each in turn and rebuilding: disabling the already-live
// COM: check in `priceTailAfter` flips %cvtB from accepted to rejected here
// COM: (corridor rises to 1185, same ceiling-exceeding value %cvtA already
// COM: gets) -- F2's fix is decisive for %cvtB in this composed context, not
// COM: just in case 37's isolated one. Disabling `siblingBlocksPeak` does
// COM: *not* change %cvtA's corridor (still 1185). The actual reason (per an
// COM: independent verification review, `review_opus55_verification_2026-
// COM: 09-27.md` V5; an earlier version of this comment blamed cross-
// COM: liveness to the shared `scf.yield`, which is not the mechanism):
// COM: `collectCorridor`'s own walk -- unrelated to either
// COM: `priceTailAfter`/`siblingBlocksPeak` or to %cvtB's own liveness --
// COM: continues past %cvtA's `%hotA` into the rest of the outer loop body
// COM: and reaches %hotB, a region-holding op that is not `forOp` itself, so
// COM: it gets priced via `regionPeakThroughOp(%hotB) + dstBytes`. %hotB's
// COM: own pre-hoist block peak is 1121 (confirmed with
// COM: -test-register-pressure), so that term alone is already
// COM: 1121 + 64 = 1185 -- the same ceiling-exceeding figure -- regardless
// COM: of what `siblingBlocksPeak` contributes for %cvtA. `siblingBlocksPeak`
// COM: still runs unconditionally on every `scf.if`-shaped candidate and
// COM: does not corrupt or regress %cvtA's (already-conservative) rejection;
// COM: case 38 remains the case that shows it is independently necessary.

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

// COM: Case 41 (#8053 follow-up, F1's "other multi-region op" question,
// COM: per `review_opus55_verification_2026-09-27.md` V6): `scf.index_switch`
// COM: is the multi-region op class where F1's undercount can actually
// COM: recur -- unlike `scf.while` (case investigated but not committed
// COM: earlier this round: for a *loop*-like `lastUse`, the back-edge rule
// COM: already keeps `%cvt` live in every one of its regions, so the
// COM: missing-`dstBytes` undercount cannot occur there at all; only the
// COM: safe, over-reject direction can). `scf.index_switch` is not a loop,
// COM: so it has the same exposure `scf.if` does: `%cvt`'s only use is in
// COM: `case 0`, a lighter `case 1` reads only `%a`, and a 17-op filler
// COM: chain (unrelated to `%cvt`/`%src`) sits in `default`. `%src` is also
// COM: read after the loop (in `%post`), so no source credit applies.
// COM: `siblingBlocksPeak` iterates `op->getRegions()` generically (no
// COM: `scf.if`-specific dispatch), so this checks that generality against a
// COM: third, structurally different region-holding op, not just against
// COM: `scf.while`'s two special-cased regions.
// COM: Measured at every GRF mode: prePeak=864, corridor=928 over 2 ops
// COM: (the `case 0` tail, empty, and the heaviest sibling case, `default`,
// COM: 864 + 64), newPoint=288; ceiling 864. Verified by hand-hoisting
// COM: through -test-register-pressure: real pre-hoist peak 864, real
// COM: post-hoist peak 928, exact match. The hoist is correctly refused.
// COM: Discriminating power confirmed by disabling `siblingBlocksPeak`
// COM: (contributing 0) and rebuilding: this case then wrongly accepts at
// COM: 864 and trips the Debug `FunctionPeakGate` invariant assert
// COM: (`freshPeak <= pendingProjection`), the same as cases 36/38.

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

// COM: Case 42 (Copilot round-4): `%cvt`'s only use is as an inner `scf.for`'s
// COM: own `iter_args` init, that inner loop itself sitting inside the outer
// COM: `scf.if` (`lastUse`). `realLastUse` resolves to the inner loop op
// COM: itself, so `nestedLast->getNumRegions() > 0`, and the precondition
// COM: added this round skips the precise `priceTailAfter`/`siblingBlocksPeak`
// COM: path entirely, falling back to `lastUse`'s own conservative
// COM: whole-region charge. Before that precondition existed, the precise
// COM: path still ran here: `priceTailAfter` walks strictly after
// COM: `nestedLast`, so it never priced the inner loop's own body, and
// COM: `siblingBlocksPeak` only covers `lastUse`'s *other* blocks, not
// COM: `nestedLast`'s nested ones -- so the inner loop's own internal peak
// COM: went unpriced by either.
// COM: Measured at every GRF mode: corridor was 673 before this round's fix,
// COM: 1505 after (the fallback's whole-region charge on `lastUse`), against
// COM: prePeak=1441 and ceiling=1441 -- the fix correctly rejects here where
// COM: the old code accepted. Checked directly whether the old accept was
// COM: actually unsound, not just incomplete: hand-running
// COM: `-test-register-pressure` on the old code's own hoisted output
// COM: measures a real peak of 1249, under the 1441 ceiling -- so in this
// COM: specific shape the old accept, while computed from a materially wrong
// COM: corridor figure, was not itself a real violation. The fix closes a
// COM: genuine gap in what gets priced either way; this case pins the
// COM: now-conservative verdict it produces, not a confirmed prior unsound
// COM: accept.

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
    // CHECK: scf.for
    %src = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked42>
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas42>) : i32 {
      // CHECK: ttg.convert_layout
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

// COM: Case 43 (#8202): case 16 with a third sibling loop. Three loops read
// COM: one %src; loop 1 carries case 16's %t chain, loops 2 and 3 are light.
// COM: Per-lane bytes as in case 16; each loop's live-in is 288.
// COM: Decided alone, last to first, identically at every mode:
// COM:   * loop 3: nothing below reads %src, net -192, but %cvt3 would land
// COM:     after %src and span loop 1: 992 + 64 = 1056 at %t4 > 992. Refused.
// COM:   * loop 2: the only later reader of %src is the refused twin %cvt3,
// COM:     which is excused, so %cvt2 is credited the same -192 and refused by
// COM:     the peak the same way (1056).
// COM:   * loop 1: both later readers are refused twins, so %cvt1 is credited
// COM:     too and hoists alone at 992 as in case 16. The merge charge prices
// COM:     loops 2 and 3 at -192 each and marks %cvt2 and %cvt3 merge-charged.
// COM: The shared-source phase then moves the refused group {%cvt3, %cvt2}
// COM: together. Both loops are merge-charged, so no loop-level gate applies,
// COM: and the peak at %t4 is 640 + 3*64 + 32 = 864 <= 992. All three land
// COM: after %src at every mode.

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

// COM: Case 44 (#8202): case 16 with an unrelated loop between the two
// COM: siblings. The middle loop has no candidate and does not read %src, but
// COM: %src and any hoisted conversion span it. Neither the twin credit nor
// COM: the grouping depends on adjacency, so the outcome is case 16's at every
// COM: mode: loop 2 refused alone by the peak at 1056, loop 1 credited past
// COM: that refused twin and hoisted alone at 992, then the singleton group
// COM: {%cvt2} hoisted with a measured peak of 800 (640 + 2*64 + 32 at %t4).

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

// COM: Case 45 (#8202): a shared-source group the measured peak refuses. Six
// COM: sibling loops read %src: loop 1 carries case 16's %t chain, loops 2-6
// COM: are light. Hoisting all six retires %src but leaves six 64-byte results
// COM: spanning loop 1: 640 + 6*64 + 32 = 1056 at %t4, above the 992 ceiling.
// COM: (Five siblings would reach exactly 992 and be accepted; six is the
// COM: smallest group this shape refuses.) No proper subset of the group
// COM: helps either, by the same measured-peak math: moving fewer than all six
// COM: still leaves %src live through loop 1, so the joint trial always
// COM: measures 1056 and is moved back regardless of group size.
// COM:
// COM: Loops 6..2 are refused alone by the peak (1056 each). Loop 1 is decided
// COM: last (loops are visited last-to-first) and sees loops 6..2's refusals
// COM: already recorded: since those refused conversions share %src's exact
// COM: (source, result type), hoisting %cvt1 would dominate and retire them
// COM: too (remove_layout_conversions merges a dominated equivalent conversion
// COM: into whatever dominates it), so %cvt1's hoist is credited as if %src
// COM: were fully retired. That credit clears both the loop-level gate and the
// COM: peak gate at 992, independent of GRF mode -- 128-GRF/default and
// COM: 256-GRF now hoist identically. The joint phase then retries
// COM: {%cvt6 .. %cvt2} as a group (since %src already has a hoisted sibling),
// COM: measures 1056, and moves it back: only %cvt1 is hoisted, same outcome
// COM: as before this credit fix at 256-GRF, now also reached at 128-GRF.
// COM:
// COM: The arithmetic, in per-lane bytes, identical at every mode (measured
// COM: with -debug-only=tritonintelgpu-hoist-layout-conversions). %src is 256,
// COM: each #dot_a result 64, %argB 32, so every loop's live-in is 288. The
// COM: peak is case 16's, at %t4: five 128-byte #dpas values 640 + %src 256 +
// COM: %cvt1 64 + %argB 32 = 992, which is also the ceiling at every mode
// COM: (prePeak exceeds both 204.8 and 409.6).
// COM:   * loop 6: the only other readers of %src are loops before it, so it
// COM:     is credited without any twin: 288 + 64 - 256 = 96, a negative delta
// COM:     (-192) the loop-level gate never vetoes. The peak refuses it: its
// COM:     corridor steps over loop 1, where %cvt1 still keeps %src live, so
// COM:     the arriving result is charged on top of loop 1's internal peak,
// COM:     992 + 64 = 1056 > 992 (corridor of 7 ops).
// COM:   * loops 5..2: each also has a reader *after* it, a later loop's
// COM:     conversion, but that conversion is an already-refused twin, so it is
// COM:     excused and each is credited the same 96. Each is then refused by
// COM:     the peak at 1056 exactly as loop 6 is (corridors of 6, 5, 4, 3
// COM:     ops). Before the twin credit these four had no credit, so 288 + 64 =
// COM:     352 >= 204.8 refused them at 128-GRF/default by the loop-level gate
// COM:     instead. That is why the STATS split for this case is five
// COM:     rejected_function_peak_exact and no rejected_pressure, not one and
// COM:     four (the split the joint-stats file pinned before the credit fix).
// COM:   * loop 1: all five later readers of %src are refused twins, so %cvt1
// COM:     is credited: 288 + 0 + (64 - 256) = 96 < 204.8. Counting the twins
// COM:     as real readers would give 288 + 64 = 352, which clears 256-GRF's
// COM:     409.6 but not 128-GRF's 204.8. The peak gate does not need the
// COM:     credit: %cvt1 is already live at %t4, so the projection stays at 992
// COM:     (corridor 608 over 2 ops, newPoint 480). Hoisted. The merge charge
// COM:     then prices each of loops 2..6 at the -192 its twin's merge is
// COM:     worth (288 - 192 = 96) and marks %cvt2..%cvt6 merge-charged.
// COM:   * shared-source phase: group {%cvt6 .. %cvt2}, all moved. Every
// COM:     member is merge-charged, so no loop-level gate applies; the
// COM:     measured peak is 640 + 6*64 + 32 = 1056 > 992, so the group is moved
// COM:     back and all five are stamped.
// COM: The MERGE run shows the premise the credit rests on: after
// COM: remove_layout_conversions, the five stamped twins are gone and every
// COM: loop's tt.dot reads the single hoisted %cvt1. remove_layout_conversions
// COM: alone keeps all six conversions in their loops, and -cse does not merge
// COM: them either (tt.no_licm makes the twins' attributes differ from
// COM: %cvt1's), so the merge needs both the hoist and that pass's
// COM: dominance-checked remat cache.

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

// COM: Case 46 (#8202): a shared-source group the loop-level gate refuses.
// COM: Two light sibling loops read %src, and so does %keep after both, so no
// COM: hoist ever retires %src from either loop: each loop's net change is +64
// COM: alone and jointly, 288 + 64 = 352. The %p chain before %src puts the
// COM: function's peak outside the loops (hand-derived: at %p4, five 128-byte
// COM: values + %arg0 256 + %argB 32 = 928), where no hoist here reaches, so
// COM: the peak gate never refuses and the loop-level gate alone decides.
// COM:   * 128-GRF/default: 352 >= 204.8 for each loop alone, and again for the
// COM:     group {%cvt2, %cvt1}, which is moved back without being measured.
// COM:     Both are stamped.
// COM:   * 256-GRF: 352 < 409.6, both hoist alone; no refusal, no group.

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

// COM: Case 47 (#8202): case 16 with %src a block argument (%arg0), the
// COM: non-goal the pass description records. With no defining op, a hoisted
// COM: conversion lands right before its own loop, so %cvt2 lands *between*
// COM: the loops and keeps %arg0 read below loop 1 even when both are moved.
// COM:   * loop 2 (every mode): retires %arg0, net -192; the new point between
// COM:     the loops holds 256 + 64 + 32 + 128 = 480 and the peak stays 992.
// COM:     Hoisted.
// COM:   * loop 1 at 128-GRF/default: no credit (%cvt2 still reads %arg0
// COM:     below), 352 >= 204.8. Refused. Its source already has a hoisted
// COM:     conversion, so the shared-source phase tries the singleton {%cvt1},
// COM:     but moving it changes nothing below loop 1: still +64, still 352,
// COM:     moved back and stamped.
// COM:   * loop 1 at 256-GRF: 352 < 409.6, and %cvt1 was already live at %t4,
// COM:     so the projection stays 992. Hoisted.

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

// COM: Case 48 (#8053): the merge charge a twin's own loop takes once a
// COM: sibling's hoist is credited past it. Case 16's two loops on %src, plus
// COM: a second source %s2 (16x16xf16 #blocked, 32 B/lane) defined between
// COM: loops 1 and 2: loop 2 also converts %s2 (%cvtX, to #dot_b, 32 B/lane)
// COM: and reads %argB; loop 3 converts %s2 again (%cvtZ); %keep reads %s2
// COM: after every loop, so no hoist ever retires %s2. Loop 2's live-in is
// COM: %src 256 + %s2 32 + %argB 32 = 320. %cvtZ converts %s2 to #dot_b48r,
// COM: not #dot_b48: its parent differs from #dpas48 only in repCluster
// COM: ([2, 1] instead of [4, 1]), and loop 3's %argA and accumulator take
// COM: that parent too, as case 50 does. Every per-lane size is unchanged
// COM: (#dot_b48r 32, #dot_a48r 64, the accumulator 128). Loop 3 starts from
// COM: a zero accumulator and %r2 is returned instead; %r2 then spans loop 3,
// COM: which no figure below sees: liveInPressure ignores a value merely
// COM: spanning a loop, and loop 3's own peak stays far under 1088. The types
// COM: differ so that %cvtX is no twin of %cvtZ: with one type, %cvtX's
// COM: refusal would be charged at once as a merge into the hoisted %cvtZ,
// COM: {%cvtX} would skip the loop-level gate, and nothing would read the
// COM: charge this case is about (that path is covered by
// COM: hoist-layout-conversions-late-refusal-merge.mlir). Decided last to
// COM: first, at 128-GRF/default:
// COM:   * loop 3: %cvtZ, +32 on a 96 live-in (%s2 + %argA); its anchor is
// COM:     %s2, below loop 1, so the peak is untouched. Hoisted.
// COM:   * loop 2: %cvt2 (-192, loop 1's %cvt1 sits before loop 2) is refused
// COM:     by the peak as in case 16; %cvtX (+32) is refused by the loop-level
// COM:     gate, 320 + 32 = 352 >= 204.
// COM:   * loop 1: %cvt1 is credited past its refused twin %cvt2 and hoisted;
// COM:     the merge charge prices loop 2 at the full -192 the merge is really
// COM:     worth (64 for %cvt1's result, less the 256 %src no longer costs
// COM:     there once %cvt2's only remaining reader is the merged value), not
// COM:     just %cvt1's own +64. netBytes[loop 2] = -192.
// COM:   * shared-source phase: group {%cvt2} is merge-charged, so its loop is
// COM:     left out of the loop-level gate; the trial is still measured, at
// COM:     896 <= 1088, and accepted. Then group {%cvtX} (%s2 already has the
// COM:     hoisted %cvtZ): 320 - 192 + 32 = 160 < 204, hoisted.
// COM: At 256-GRF, %cvtX clears the loop-level gate in the first phase (352 <
// COM: 409.6) and is hoisted there, so the charge is never consulted for it.
// COM: All four conversions land outside their loops at every GRF mode; none
// COM: is stamped. (An earlier, since-corrected version of this charge flat-
// COM: debited loop 2 only +64 instead of the full -192, which double-counted
// COM: %cvt2's 64 bytes against group {%cvt2}'s own delta and left %cvtX
// COM: refused at 224 instead of hoisted at 160. With group {%cvt2}'s
// COM: merge-charged skip in place, the same flat +64 gives 320 + 64 + 32 =
// COM: 416 >= 204, refused all the same.) This case pins that correction
// COM: only: with no merge charge at all, the accepted group {%cvt2} would
// COM: price loop 2 at the same -192 itself, before {%cvtX} is tried, and
// COM: the outcome would be identical. Case 51 is the shape whose outcome
// COM: needs the charge.
// COM: (Predicted from static analysis, not yet confirmed against a build:
// COM: the retyping of %cvtZ and loop 3 postdates this case's last confirmed
// COM: trace. Every figure above is predicted to be unchanged from it.)

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

// COM: Case 49 (#8053): the cascade the twin credit was written for, from
// COM: _attn_bwd's backward loop: loop 2's K conversion is refused, loop 1's
// COM: K conversion is credited past it and hoisted, and that credit is what
// COM: frees loop 1's budget for its *own* V conversion, a second candidate
// COM: on a different source. Case 16's two loops on %src, plus %vs (16x16xf16
// COM: #blocked, 32 B/lane) converted in loop 1 by %cvtV (to #dot_b, 32) and
// COM: read again by %keep after both loops, so no hoist ever retires %vs and
// COM: %cvtV always costs its full +32. Loop 2 reads only %src and constants,
// COM: which are rematerializable and so free. Loop 1's live-in is %src 256 +
// COM: %vs 32 + %argB 32 = 320; the peak is at %t4: 640 + %src 256 + %cvt1 64
// COM: + %cvtV 32 + %vs 32 + %argB 32 = 1056, also the ceiling at every mode.
// COM: Decided last to first, at every mode:
// COM:   * loop 2: %cvt2 is credited (-192, nothing after loop 2 reads %src)
// COM:     but refused by the peak: hoisted, its result spans loop 1, 1056 +
// COM:     64 = 1120 > 1056 (corridor of 6 ops).
// COM:   * loop 1: %cvt2 is a refused twin of %cvt1 (same %src, same #dot_a),
// COM:     so %cvt1 is credited, delta -192, and goes first as the cheapest:
// COM:     320 - 192 = 128, peak stays 1056 (corridor 640 over 3 ops,
// COM:     newPoint 512). Hoisted; netBytes[loop 1] = -192, and the merge
// COM:     charge prices loop 2 at -192 too and marks %cvt2 merge-charged.
// COM:     Then %cvtV: 320 - 192 + 32 = 160 < 204.8, peak 1056 (corridor 672
// COM:     over 2 ops, newPoint 544). Hoisted.
// COM:   * shared-source phase: group {%cvt2} (%src already hoisted). It is
// COM:     merge-charged, so no loop-level gate applies; measured peak 640 +
// COM:     2*64 + %cvtV 32 + %vs 32 + %argB 32 = 864 <= 1056. Hoisted.
// COM: All three land above loop 1 and none is stamped. Without the credit,
// COM: %cvt1 would cost +64, so %cvtV (+32) would be decided first at 320 + 32
// COM: = 352 >= 204.8 and refused, and %cvt1 after it at 384. The shared-source
// COM: phase would then rescue %cvt1 together with %cvt2, but never %cvtV: %vs
// COM: has one refusal and no hoisted conversion, so it is not regrouped. Case
// COM: 50 is that counterfactual, measured: the same function with only
// COM: %cvt2's result type changed, so the twin credit cannot fire.
// COM: The cascade is load-bearing at 128-GRF/default only: at 256-GRF %cvtV
// COM: clears 352 < 409.6 without any credit, so the GRF256 output is the
// COM: same here and in case 50.

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

// COM: Case 50 (#8053): no twin credit for the same source converted to a
// COM: *different* result type. Case 49 with one change: loop 2's %cvt2 converts
// COM: %src to #dot_a50r, whose parent differs from #dpas50 only in repCluster
// COM: ([2, 1] instead of [4, 1]), and loop 2's constants and accumulator take
// COM: that parent too. #dot_a50r is still 64 B/lane, so every byte figure
// COM: below is case 49's; only the type check in hoistRetiresSource's
// COM: isUnenforceableTwin sees a difference. remove_layout_conversions keys its
// COM: remat cache on (value, encoding) and would never merge these two, which
// COM: is why the credit must not fire. At 128-GRF/default:
// COM:   * loop 2: as in case 49, refused by the peak at 1120 > 1056.
// COM:   * loop 1: %cvt2 is not a twin of %cvt1, so it counts as a real reader
// COM:     of %src after loop 1 and %cvt1 earns no credit: +64. The cheapest
// COM:     candidate is now %cvtV, 320 + 32 = 352 >= 204.8, refused; then %cvt1,
// COM:     320 + 64 = 384 >= 204.8, refused. Both by the loop-level gate.
// COM:   * shared-source phase: it groups refusals by source alone, regardless
// COM:     of type, so {%cvt2, %cvt1} is moved together. Both loops then retire
// COM:     %src (-192 each), and the measured peak is 864 <= 1056, so both are
// COM:     hoisted. %cvtV is never regrouped (one refusal, nothing of %vs
// COM:     hoisted) and stays stamped.
// COM: So %cvt1 ends up hoisted here just as in case 49, by a type-agnostic
// COM: mechanism that has nothing to do with the credit, and its placement
// COM: cannot tell the two cases apart. The stamped %cvtV can: it is refused
// COM: in the one-at-a-time phase, which the shared-source phase never
// COM: revisits for %vs, and only because %cvt1 got no credit there.
// COM: At 256-GRF: %cvtV goes first, 352 < 409.6, and is hoisted (peak 1056,
// COM: corridor 608 over 3 ops, newPoint 480); %cvt1 is refused at 320 + 32 +
// COM: 64 = 416 >= 409.6; the group {%cvt2, %cvt1} is then hoisted at 864.
// COM: Nothing is stamped, the same as case 49 at 256-GRF.
// COM: The MERGE run shows that remove_layout_conversions leaves both
// COM: conversions of %src in place: one value, two encodings, two cache keys.

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

// COM: Case 51 (#8053): the shape whose outcome needs the merge charge, which
// COM: case 48's does not (see there). Case 48 plus four more light loops
// COM: (2b..2e) between loops 2 and 3, each converting %src to #dot_a again,
// COM: so the refused twins of %cvt1 are five and their group fails the
// COM: shared-source phase, as case 45's does. The charge %cvt1's hoist makes
// COM: is then the only thing that prices loop 2's merge before {%cvtX} is
// COM: tried. %cvtZ and loop 3 are retyped as in case 48 (to #dot_b51r and
// COM: #dpas51r, %r2e returned instead of feeding loop 3) and for the same
// COM: reason: with one type, %cvtX's refusal would itself be charged as a
// COM: merge into %cvtZ, and {%cvtX} would skip the loop-level gate.
// COM: Per-lane bytes as in case 48: %src 256, each #dot_a result 64, %s2
// COM: and each #dot_b result 32, %argB 32, %argA 64. Loop 1's live-in is
// COM: %src + %argB = 288, loop 2's is %src + %s2 + %argB = 320, loops
// COM: 2b..2e's 288, loop 3's %s2 + %argA = 96. The peak is case 48's, at %t4:
// COM: 640 + %src 256 + %cvt1 64 + %argB 32 + %argS2 32 + %argA 64 = 1088,
// COM: also the ceiling at every mode.
// COM: Decided last to first, at 128-GRF/default:
// COM:   * loop 3: %cvtZ, 96 + 32 = 128 < 204. Hoisted, as in case 48.
// COM:   * loops 2e..2b: each credited, the later readers of %src being
// COM:     refused twins (none for 2e), 288 - 192 = 96, and each refused by the
// COM:     peak: its result would span loop 1, where %src is still live, 1088
// COM:     + 64 = 1152 > 1088.
// COM:   * loop 2: %cvt2 likewise (-192, refused at 1152); %cvtX, +32 (%keep
// COM:     reads %s2 below), 320 + 0 + 32 = 352 >= 204. Refused.
// COM:   * loop 1: %cvt1 is credited past all five twins, 288 - 192 = 96,
// COM:     peak 1088, hoisted. The merge charge prices loops 2 and 2b..2e at
// COM:     -192 each: netBytes[loop 2] = -192.
// COM:   * shared-source phase, groups in the order of their first refusal:
// COM:     {%cvt2e, %cvt2d, %cvt2c, %cvt2b, %cvt2}, all merge-charged, so no
// COM:     loop-level gate; measured 640 + 6*64 + 32 + 32 + 64 = 1152 > 1088,
// COM:     moved back, all five stamped. Then {%cvtX} (%s2 already has the
// COM:     hoisted %cvtZ): loop 2 at 320 - 192 + 32 = 160 < 204, measured 1088
// COM:     <= 1088. Hoisted.
// COM: Without the merge charge, loop 2's figure would still be 0 when
// COM: {%cvtX} is tried, since the group that would have priced its merge
// COM: failed: 320 + 0 + 32 = 352 >= 204, and %cvtX would be stamped too. A
// COM: flat +64 charge in its place (case 48's earlier bug) gives 320 + 64 +
// COM: 32 = 416 >= 204, and stamps %cvtX the same way.
// COM: At 256-GRF, %cvtX clears the loop-level gate in the first phase (352 <
// COM: 409.6) and everything else is decided as above, so the output is the
// COM: same at every mode; only 128-GRF/default exercise the charge.
// COM: (Predicted from static analysis, not yet confirmed against a build:
// COM: the retyping of %cvtZ and loop 3 postdates this case's last confirmed
// COM: trace. Every figure above is predicted to be unchanged from it.)

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

// COM: Case 52 (#8053): no twin credit for an FMA dot's operands. Case 45's six
// COM: loops with the dot operands' parent a #ttg.blocked layout instead of
// COM: #dpas, and constant B operands. remove_layout_conversions never
// COM: rematerializes backward into a dot operand whose parent is blocked, so
// COM: it would leave every refused twin in its loop reading %src: none of
// COM: them is excused, and each counts as the real reader it stays.
// COM: Per-lane bytes: %src 256 as in case 45; a #dot_a52 result 64 (2 rows of
// COM: the full K of 16 per lane); a 128x16xf32 #fma52 value 128; %zB a
// COM: constant, rematerializable and so free. Every loop's live-in is 256.
// COM: The peak is at %t4: 640 + %src 256 + %cvt1 64 = 960, also the ceiling
// COM: at every mode. Decided last to first, at 128-GRF/default:
// COM:   * loop 6: nothing below reads %src, so it is credited, 256 + 64 -
// COM:     256 = 64, but refused by the peak: its result would span loop 1,
// COM:     where %src is still live, 960 + 64 = 1024 > 960.
// COM:   * loops 5..2: each has a later refused conversion of %src to the same
// COM:     type, not excused, so no credit: 256 + 64 = 320 >= 204.8. Refused.
// COM:   * loop 1: likewise, 320 >= 204.8. Refused.
// COM:   * shared-source phase: group {%cvt6 .. %cvt1}, all moved. Every loop
// COM:     then retires %src, -192, but the measured peak is 640 + 6*64 =
// COM:     1024 > 960, so the group is moved back and all six are stamped.
// COM: Under a twin credit blind to the parent layout, loops 5..2 would be
// COM: credited and refused by the peak instead, and loop 1 credited, 256 -
// COM: 192 = 64, and hoisted at 960, leaving its stamped twins reading %src
// COM: inside their loops after remove_layout_conversions, so the credit
// COM: would have been unsound for loop 1.
// COM: At 256-GRF: loops 5..2 clear the loop-level gate (320 < 409.6) and are
// COM: refused by the peak at 1024 like loop 6; loop 1 clears it too and is
// COM: hoisted (projected 960, %cvt1 already being live at %t4); the group
// COM: {%cvt6 .. %cvt2} measures 1024 and is moved back.

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

// COM: Case 53 (#8053): no twin credit past a twin the hoist would not
// COM: dominate. %arg0 is a block argument, so a hoisted conversion of it
// COM: lands just above its own loop rather than below a defining op. Loop 1
// COM: sits in an scf.if and loop 2 after it, both in an outer loop: %cvt1
// COM: would land inside the scf.if, which does not dominate loop 2, so
// COM: remove_layout_conversions would leave a refused %cvt2 in place, reading
// COM: %arg0. %cvt2 is not excused, and counts as the real reader it stays.
// COM: Per-lane bytes: %arg0 256, each #dot_a result 64, %argB 32, each
// COM: #dpas value 128, %cond 1. Both inner loops' live-in is %arg0 + %argB =
// COM: 288. The peak is at loop 2's %t4: five #dpas values 640 + %arg0 256 +
// COM: %argB 32 + %cond 1 = 929 (all three are read in the outer loop, so
// COM: they are live through all of it), also the ceiling at every mode.
// COM: Decided last to first, identically at every mode for loop 2:
// COM:   * loop 2: %cvt1 sits before loop 2, so %cvt2 is credited, 288 + 64 -
// COM:     256 = 96, but %arg0 is also read in the outer loop, whose back edge
// COM:     the projection does not model, so it charges %cvt2's result with no
// COM:     credit wherever it becomes newly live: 929 + 64 = 993 > 929 at %t4
// COM:     (conservatively charged). Refused.
// COM:   * loop 1 at 128-GRF/default: %cvt2 is not excused, so no credit: 288
// COM:     + 64 = 352 >= 204.8. Refused.
// COM:   * shared-source phase: group {%cvt2, %cvt1}. Moved, %cvt2 lands
// COM:     between the scf.if and loop 2, still not dominated by %cvt1 and
// COM:     still reading %arg0 after loop 1: loop 1 at 352 again, moved back,
// COM:     both stamped.
// COM:   * loop 1 at 256-GRF: 352 < 409.6, and the projection stays at 929 (its
// COM:     corridor holds only loop 1, light, and its yield). Hoisted, to the
// COM:     top of the scf.if's then block; the group {%cvt2} then measures 993
// COM:     and is moved back.
// COM: With the twin credit, %cvt1 would be credited, 288 - 192 = 96, and
// COM: hoisted at every mode, although %arg0 stays read below loop 1.

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

// COM: Case 54 (#8053): no twin credit past a twin the loop-level gate
// COM: refused. When a conversion's result is per-lane fatter than its source,
// COM: the merge leaves the twin's loop exactly as hoisting the twin would,
// COM: which is what that gate refused, so the credit must not override it.
// COM: Per-lane bytes: %s and %w (16x16xf16, one element per lane in each of
// COM: four rows) 8 each, a #dot_b result 32, each #dot_a argument 64. Loop 1's
// COM: live-in is %a1 + %a2 + %argB + %s + %w = 64 + 64 + 32 + 8 + 8 = 176;
// COM: loop 2's is %a1 + %a2 + %a3 + %s = 200. The peak is at loop 1's %d:
// COM: %acc 128 + %d 128 + %a1, %a2, %a3 192 + %h 32 + %argB 32 + %s 8 + %w 8
// COM: = 528, also the ceiling at every mode.
// COM: Decided last to first, at 128-GRF/default:
// COM:   * loop 2: %t, credited (%h sits before loop 2), 32 - 8 = +24, 200 + 24
// COM:     = 224 >= 204.8. Refused by the loop-level gate.
// COM:   * loop 1: %t would merge into a hoisted %h, but that merge leaves loop
// COM:     2 at the same 224, so %t is a real reader after loop 1 and %h gets
// COM:     no credit: 176 + 32 = 208 >= 204.8. Refused.
// COM:   * shared-source phase: group {%t, %h}, loop 2 at 224 again, moved
// COM:     back, both stamped.
// COM: At 256-GRF: %t clears the loop-level gate (224 < 409.6) but is refused
// COM: by the peak, its result spanning loop 1, 528 + 32 = 560 > 528. %h is
// COM: credited past it, the merge leaving loop 2 at 224 < 409.6: 176 + 24 =
// COM: 200, and hoisted at 528. The group {%t} measures 552 > 528 and is
// COM: moved back.
// COM: With an unconditional twin credit, %h would also be credited at
// COM: 128-GRF/default, 176 + 24 = 200 < 204.8, and hoisted, and after the
// COM: merge loop 2's live-in would be the 224 its own gate refused.

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
