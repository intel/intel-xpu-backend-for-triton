// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128 corridor-op-cap=5" | FileCheck %s

// COM: Case: a twin refused only *after* the hoist it merges into. Loops are
// COM: decided last to first, so a physically later loop's conversion can be
// COM: hoisted before a physically earlier loop's same-pair conversion is even
// COM: weighed. When that earlier one is then refused, remove_layout_conversions
// COM: still replaces it with the hoisted result, and its loop is now charged
// COM: for that merge at the refusal, so a later shared-source group there
// COM: sees the corrected figure. The mirror image of
// COM: hoist-layout-conversions-uncharged-merge.mlir, where the refusal comes
// COM: first and the hoist charges it.
// COM:
// COM: PREDICTED FROM STATIC ANALYSIS, NOT YET CONFIRMED AGAINST A BUILD.
// COM: Every number below is hand-derived from the pass's source and from the
// COM: confirmed trace of hoist-layout-conversions-uncharged-merge.mlir, whose
// COM: types and loop bodies this case reuses with the loops reordered. The
// COM: lines marked (exact) follow from per-lane type sizes and liveInPressure
// COM: alone; the others depend on whole-function pressure figures and need a
// COM: real -debug-only=tritonintelgpu-hoist-layout-conversions trace.
// COM: Kept apart from hoist-layout-conversions.mlir for the same reason that
// COM: file is: corridor-op-cap=5 is what makes %y's phase-1 refusal a
// COM: conservative fallback, and adding that RUN line to a shared file would
// COM: rerun every other case there under it.
// COM:
// COM: Per-lane bytes (grf-mode=128: budget 256, threshold 204): %s
// COM: (32x16xf16 #blk) 16, a #dot_b result of it 64; %q and %arg2 (16x16xf16
// COM: #blk) 8 each, a #dot_b or #dot_br result of %q 32; %arg3 128; each
// COM: 128x16xf32 value 128. Constants are rematerializable and free. Loop
// COM: live-ins (exact): loop 2 %s + %q + %arg2 + %arg3 = 160, loop H %s = 16,
// COM: loop 3 %q = 8. The function peak is the tail chain's, eight 128-byte
// COM: values live at %p7: 1024, also the ceiling.
// COM:
// COM: Predicted trace, in decision order:
// COM:   L3 %z  Hoisting ...: liveIn=8 + alreadyHoisted=0 + thisHoist=24 = 32
// COM:          (exact; %y in loop 2 is the only other reader of %q and sits
// COM:          before loop 3). Peak: projected 1024 (prePeak=1024, corridor
// COM:          over 4 ops, newPoint=320), within ceiling 1024.
// COM:   LH %h  Hoisting ...: liveIn=16 + alreadyHoisted=0 + thisHoist=48 = 64
// COM:          (exact; %t in loop 2 sits before loop H, so %s retires: 64 -
// COM:          16). Peak: projected 1024 (prePeak=1024, corridor over 5 ops,
// COM:          its largest term loop 2's internal peak, about 576, + 64,
// COM:          newPoint about 352), within ceiling 1024.
// COM:   L2 %y  Skipping hoist: projected function peak 1056 B/lane (max of
// COM:          prePeak=1024 corridor=1056 over 0 ops, newPoint=384) exceeds
// COM:          ceiling 1024 B/lane (conservatively charged). %y is cheapest
// COM:          (24, exact) and passes the loop-level gate at 184 < 204, but its
// COM:          corridor (%z, loop 2, then %t, %d, %fx, ...) exceeds 5 ops.
// COM:   L2 %t  Skipping hoist: liveIn=160 + alreadyHoisted=0 + thisHoist=48
// COM:          = 208 B/lane exceeds 80% of budget=256 B/lane (exact).
// COM:   L2 %t  Refused conversion merges into an earlier-decided loop's
// COM:          hoist: its loop is now at alreadyHoisted=48 B/lane (exact).
// COM:   phase 2 {%y}: Skipping joint hoist of 1 conversion(s): loop liveIn=
// COM:                 160 + alreadyHoisted=48 + thisHoist=24 = 232 B/lane
// COM:                 exceeds 80% of budget=256 B/lane (exact).
// COM:   phase 2 {%t}: Function peak allows joint hoist: measured 1024 B/lane
// COM:                 within ceiling 1024 B/lane; Hoisting jointly a group
// COM:                 of 1 conversion(s) sharing a source.
// COM:
// COM: The fix: when %t is refused, %h is already hoisted (loop H, decided
// COM: before loop 2), with %t's exact source and result type, and %s has a
// COM: defining op, so mergesIntoHoist holds without any position condition.
// COM: collectMergingTwins never saw %t (it was no refusal when %h hoisted);
// COM: chargeMergeIntoPriorHoist charges it now, through chargeMergedTwins:
// COM: netBytes[loop 2] = 0 + 48 = 48, %t.mergeCharged = true. This is the
// COM: dst > src shape (64 > 16), where an uncharged merge is an undercharge:
// COM: after the merge, loop 2 reads %h's 64 bytes instead of %s's 16, the
// COM: same +48 %t's refusal said it could not afford.
// COM:
// COM: Phase 2 then tries {%y} first (refused first in phase 1): 160 + 48 +
// COM: 24 = 232 >= 204, refused by the loop-level gate, so %y stays in loop
// COM: 2, stamped tt.no_licm. Without the fix this read 160 + 0 + 24 = 184 <
// COM: 204, and the measured peak (loop 2's internal peak plus %y's 24 net,
// COM: well under the tail's 1024) would accept it, hoisting %y. That stamp
// COM: is this case's discriminator.
// COM:
// COM: %t's own fate is not pinned. With the fix it is merge-charged, so its
// COM: {%t} trial is gated only by the measured peak, predicted to accept it
// COM: (it then lands right after %s, beside %h). Without the fix the order
// COM: of outcomes flips: {%y} is hoisted, and {%t} is then gated at 160 + 24
// COM: + 48 = 232 >= 204 and stamped. The checks below match either fate of
// COM: %t, so they fail without the fix only through %y.

#blk = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dpasr = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dot_ar = #ttg.dot_op<{opIdx = 0, parent = #dpasr, kWidth = 1}>
#dot_br = #ttg.dot_op<{opIdx = 1, parent = #dpasr, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @late_refusal_merge_now_charged
  tt.func @late_refusal_merge_now_charged(%arg0: tensor<32x16xf16, #blk>, %arg1: tensor<16x16xf16, #blk>, %arg2: tensor<16x16xf16, #blk>,
      %arg3: tensor<128x32xf16, #dot_a>, %arg4: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %zA32 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #dot_a>
    %zA16 = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_a>
    %zAr = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_ar>
    %zCr = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpasr>
    // COM: %h is hoisted right after %s (and %t too, in phase 2, if its
    // COM: predicted trial holds); the first unstamped match is enough.
    // CHECK: %[[S:.*]] = arith.addf %arg0, %arg0
    // CHECK: ttg.convert_layout %[[S]] : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %s = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blk>
    // COM: %z is hoisted right after %q.
    // CHECK: %[[Q:.*]] = arith.addf %arg1, %arg1
    // CHECK: ttg.convert_layout %[[Q]] : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %q = arith.addf %arg1, %arg1 : tensor<16x16xf16, #blk>
    // CHECK: scf.for
    %r2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %arg4) -> (tensor<128x16xf32, #dpas>) : i32 {
      %t = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %arg3, %t, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      %fx = arith.addf %arg2, %arg2 : tensor<16x16xf16, #blk>
      %x1 = arith.addf %d, %d : tensor<128x16xf32, #dpas>
      %x2 = arith.addf %x1, %d : tensor<128x16xf32, #dpas>
      %x3 = arith.addf %x2, %d : tensor<128x16xf32, #dpas>
      %x4 = arith.addf %x3, %d : tensor<128x16xf32, #dpas>
      // COM: The case's own point: %y stays refused, since loop 2's netBytes
      // COM: already holds %t's merge into %h, charged when %t was refused.
      // CHECK: ttg.convert_layout %[[Q]] {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
      %y = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_b>
      %e = tt.dot %zA16, %y, %x4, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %e : tensor<128x16xf32, #dpas>
    }
    %r1 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %h = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %zA32, %h, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d : tensor<128x16xf32, #dpas>
    }
    %r3 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zCr) -> (tensor<128x16xf32, #dpasr>) : i32 {
      %z = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_br>
      %d = tt.dot %zAr, %z, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_ar> * tensor<16x16xf16, #dot_br> -> tensor<128x16xf32, #dpasr>
      scf.yield %d : tensor<128x16xf32, #dpasr>
    }
    %p1 = arith.addf %r1, %r1 : tensor<128x16xf32, #dpas>
    %p2 = arith.addf %p1, %r1 : tensor<128x16xf32, #dpas>
    %p3 = arith.addf %p2, %p1 : tensor<128x16xf32, #dpas>
    %p4 = arith.addf %p3, %p2 : tensor<128x16xf32, #dpas>
    %p5 = arith.addf %p4, %p3 : tensor<128x16xf32, #dpas>
    %p6 = arith.addf %p5, %p4 : tensor<128x16xf32, #dpas>
    %p7 = arith.addf %p6, %p1 : tensor<128x16xf32, #dpas>
    %p8 = arith.addf %p7, %p2 : tensor<128x16xf32, #dpas>
    %p9 = arith.addf %p8, %p3 : tensor<128x16xf32, #dpas>
    %p10 = arith.addf %p9, %p4 : tensor<128x16xf32, #dpas>
    %p11 = arith.addf %p10, %p5 : tensor<128x16xf32, #dpas>
    %p12 = arith.addf %p11, %p6 : tensor<128x16xf32, #dpas>
    %p13 = arith.addf %p12, %p7 : tensor<128x16xf32, #dpas>
    tt.return %p13, %r3, %p8 : tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>
  }
}
