// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128 corridor-op-cap=5" | FileCheck %s

// COM: Case: a twin merges away for free when its sibling hoists on its own
// COM: uncredited merits, and the merge is now charged to the twin's loop so
// COM: a later shared-source group there sees the corrected figure.
// COM:
// COM: CONFIRMED against a real triton-opt build: the numbers below are
// COM: quoted verbatim from a real -debug-only=tritonintelgpu-hoist-layout-
// COM: conversions trace and the actual triton-opt output of this exact RUN
// COM: line, not derived by hand.
// COM: This file is deliberately separate from hoist-layout-conversions.mlir
// COM: and hoist-layout-conversions-corridor-cap.mlir: corridor-op-cap=5 is
// COM: needed to reproduce this scenario (without it, %y's own phase-1
// COM: refusal might take the precise corridor path instead of the fallback
// COM: one and land at a different number, or not refuse at all), and adding
// COM: a new RUN line with that option to either shared file would run it
// COM: against every other case there too, each needing its own
// COM: re-verification; kept separate instead.
// COM:
// COM: Real trace (grf-mode=128 corridor-op-cap=5, quoted
// COM: verbatim from the real build):
// COM:   L3 %z  Hoisting ...: liveIn=8 + alreadyHoisted=0 + thisHoist=24 = 32
// COM:   L2 %y  Skipping hoist: projected function peak 1056 B/lane (max of
// COM:          prePeak=1024 corridor=1056 over 0 ops, newPoint=320) exceeds
// COM:          ceiling 1024 B/lane (conservatively charged)
// COM:   L2 %t  Skipping hoist: liveIn=160 + alreadyHoisted=0 + thisHoist=48
// COM:          = 208 B/lane exceeds 80% of budget=256 B/lane
// COM:   L1 %h  No twin credit: merging would leave its loop at liveIn=160 +
// COM:          alreadyHoisted=0 + thisHoist=48 = 208 B/lane, past 80% of
// COM:          budget=256 B/lane
// COM:   L1 %h  Hoisting ...: liveIn=16 + alreadyHoisted=0 + thisHoist=64 = 80
// COM:   phase 2 {%y}: Skipping joint hoist of 1 conversion(s): loop liveIn=
// COM:                 160 + alreadyHoisted=48 + thisHoist=24 = 232 B/lane
// COM:                 exceeds 80% of budget=256 B/lane
// COM:   phase 2 {%t}: Function peak allows joint hoist: measured 1024 B/lane
// COM:                 within ceiling 1024 B/lane; Hoisting jointly a group
// COM:                 of 1 conversion(s) sharing a source
// COM: None of %h's, %z's or %t's own phase-1 decisions depend on this fix:
// COM: it only changes what loop 2's netBytes reads as by the time phase 2
// COM: runs. Loops are decided last-to-first (loop 3, then loop 2, then loop
// COM: 1 last), so %h and %z are decided and hoisted exactly as the trace
// COM: shows, before this fix's own charging code ever runs.
// COM:
// COM: The fix: when %h hoists (loop 1, decided last), collectMergingTwins
// COM: finds %t as a structural twin (same source %s, same result type,
// COM: mergesIntoHoist holds since %s has a defining op) regardless of
// COM: mergeFits having just refused %h's own credit past it. chargeMergedTwins
// COM: then charges loop 2's netBytes 48 (hoistDeltaBytes(%t, loop2, ...)):
// COM: netBytes[loop2] = 0 + 48 = 48, and %t.mergeCharged = true. This
// COM: happens during phase 1 (loop 1's own decision), strictly before
// COM: phase 2 starts for any group.
// COM:
// COM: Phase 2 then tries {%y} first (refused first, chronologically, in
// COM: phase 1), now against the corrected baseline:
// COM:   160 (liveIn) + 48 (netBytes[loop2], was 0 before this fix) + 24
// COM:   (%y's own delta, unchanged by this fix) = 232 >= 204 (threshold,
// COM:   80% of 256 B/lane at grf-mode=128) -> refused by the loop-level
// COM:   gate, before the peak gate is even asked. Before this fix this read
// COM:   184 < 204 and was accepted, physically hoisting %y.
// COM: %y therefore stays inside loop 2, stamped tt.no_licm. This is the
// COM: case's own discriminator: it is the one observable that depends on
// COM: this fix existing.
// COM:
// COM: %t's own fate: because %t is mergeCharged, reconsiderSharedSources'
// COM: own per-refusal loop-level skip excludes it from the loop-level gate
// COM: entirely for its own {%t} group trial, leaving only the whole-function
// COM: peak gate's measurement to pass or fail it. That measurement accepts
// COM: it (measured 1024 within the 1024 ceiling), so %t is physically
// COM: hoisted alongside %h: in the real build's output, %t's conversion
// COM: (the first ttg.convert_layout of %s, placed before loop 1 together
// COM: with %h's own) carries no tt.no_licm stamp. No CHECK line below pins
// COM: this specifically -- the %[[S]] pattern just below only requires one
// COM: unstamped match, which %h alone would already satisfy -- so this
// COM: observation is not itself guarded by this test; it is reported here
// COM: as a confirmed-once fact about the real behavior, not as a locked
// COM: invariant. This matches reconsiderSharedSources' own doc comment,
// COM: which treats a
// COM: mergeCharged-only group's trial verdict as a don't-care ("though it
// COM: merges downstream whatever the verdict"), since %t merges away via
// COM: remove_layout_conversions either way -- it happens to also be
// COM: physically moved here, but that is this group's own measurement, not
// COM: something the fix depends on.

#blk = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dpasr = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dot_ar = #ttg.dot_op<{opIdx = 0, parent = #dpasr, kWidth = 1}>
#dot_br = #ttg.dot_op<{opIdx = 1, parent = #dpasr, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @uncharged_merge_now_charged
  tt.func @uncharged_merge_now_charged(%arg0: tensor<32x16xf16, #blk>, %arg1: tensor<16x16xf16, #blk>, %arg2: tensor<16x16xf16, #blk>,
      %arg3: tensor<128x32xf16, #dot_a>, %arg4: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %zA32 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #dot_a>
    %zA16 = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_a>
    %zAr = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_ar>
    %zCr = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpasr>
    // COM: %h hoists on its own uncredited merits (delta 64, no retirement
    // COM: credit, since %t is a real reader from loop 1's perspective once
    // COM: mergeFits refuses the twin credit): liveIn=16 + 0 + 64 = 80 < 204.
    // CHECK: %[[S:.*]] = arith.addf %arg0, %arg0
    // CHECK: ttg.convert_layout %[[S]] : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %s = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blk>
    // CHECK: scf.for
    %r1 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %arg4) -> (tensor<128x16xf32, #dpas>) : i32 {
      %h = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %zA32, %h, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d : tensor<128x16xf32, #dpas>
    }
    // COM: %z hoists on its own merits too (unaffected by this fix either
    // COM: way: it is decided before %h, in loop 3, the first loop decided).
    // CHECK: %[[Q:.*]] = arith.addf %arg1, %arg1
    // CHECK: ttg.convert_layout %[[Q]] : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %q = arith.addf %arg1, %arg1 : tensor<16x16xf16, #blk>
    // CHECK: scf.for
    %r2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %t = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %arg3, %t, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      %fx = arith.addf %arg2, %arg2 : tensor<16x16xf16, #blk>
      %x1 = arith.addf %d, %d : tensor<128x16xf32, #dpas>
      %x2 = arith.addf %x1, %d : tensor<128x16xf32, #dpas>
      %x3 = arith.addf %x2, %d : tensor<128x16xf32, #dpas>
      %x4 = arith.addf %x3, %d : tensor<128x16xf32, #dpas>
      // COM: The case's own point: %y now correctly stays refused, since
      // COM: loop 2's netBytes already reflects %t's merge (charged when %h
      // COM: hoisted, above) by the time this is decided.
      // CHECK: ttg.convert_layout %[[Q]] {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
      %y = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_b>
      %e = tt.dot %zA16, %y, %x4, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %e : tensor<128x16xf32, #dpas>
    }
    %r3 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zCr) -> (tensor<128x16xf32, #dpasr>) : i32 {
      %z = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_br>
      %d = tt.dot %zAr, %z, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_ar> * tensor<16x16xf16, #dot_br> -> tensor<128x16xf32, #dpasr>
      scf.yield %d : tensor<128x16xf32, #dpasr>
    }
    %p1 = arith.addf %r2, %r2 : tensor<128x16xf32, #dpas>
    %p2 = arith.addf %p1, %r2 : tensor<128x16xf32, #dpas>
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
