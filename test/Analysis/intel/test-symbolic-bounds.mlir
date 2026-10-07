// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-symbolic-bounds -verify-diagnostics=only-expected | FileCheck %s

// COM: The prover's rules are unit-tested directly in
// COM: unittest/Analysis/SymbolicBoundsTest.cpp. The sections here are the
// COM: ones that need a whole module: two realistic loops end to end, the
// COM: element-correspondence case that must never be refuted, and a TTGIR
// COM: layout conversion, which a gtest string cannot carry.
// COM:
// COM: `-verify-diagnostics=only-expected` ignores unannotated remarks, so a
// COM: comparison with no `expected-remark` above it asserts nothing; only the
// COM: annotated ones are pinned. There is no `scf.for` remark: the prover
// COM: has no trip-count API.

// COM: The inductor reduction shape: `r + lane < rnumel` over
// COM: a loop `0 to rnumel step 64`. Unprovable as written - the last
// COM: iteration's tail lanes are masked off - and provable exactly when the
// COM: loop ends on an iteration boundary, the exact-loop-end condition on
// COM: the loop's own upper bound.

// CHECK-LABEL: tt.func @reduction_loop
module {
  tt.func @reduction_loop(%ptr: !tt.ptr<f32>, %rnumel: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
    scf.for %r = %c0 to %rnumel step %c64 : i32 {
      %rs = tt.splat %r : i32 -> tensor<64xi32>
      %idx = arith.addi %rs, %lane : tensor<64xi32>
      // expected-remark@+1 {{verdict: Conditional{arg1 divisible by 64}}}
      %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: The tutorial-03 K loop: `lane < K - 64*k` over a loop
// COM: whose upper bound is `cdiv(K, 64)`. Three conditions, in the fixed
// COM: order facts, preconditions, guards:
// COM:   - `arg0 divisible by 64` is the exact-cdiv candidate;
// COM:   - `arg0 >= 0` is the precondition of the division facts, which hold
// COM:     only for a non-negative dividend. It does not change lo(d), and
// COM:     dropping it would prove the mask for a negative K;
// COM:   - `arg0 <= 2147483584` is the wrap guard on the cdiv numerator
// COM:     `K + 63`, which is INT32_MAX for K near the top of i32 and makes
// COM:     the quotient, and so the proof, wrong.
// COM: The matching `K + 63 >= INT32_MIN` guard is omitted because K's own
// COM: i32 range already implies it.

// CHECK-LABEL: tt.func @cdiv_k_loop
module {
  tt.func @cdiv_k_loop(%K: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c63 = arith.constant 63 : i32
    %c64 = arith.constant 64 : i32
    %num = arith.addi %K, %c63 : i32
    %q = arith.divsi %num, %c64 : i32
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    scf.for %k = %c0 to %q step %c1 : i32 {
      %k64 = arith.muli %k, %c64 : i32
      %rem = arith.subi %K, %k64 : i32
      %rs = tt.splat %rem : i32 -> tensor<64xi32>
      // expected-remark@+1 {{verdict: Conditional{arg0 divisible by 64; arg0 >= 0; arg0 <= 2147483584}}}
      %mask = arith.cmpi slt, %lane, %rs : tensor<64xi32>
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: One `make_range` reaching a comparison through two different
// COM: `expand_dims` axes: `r[:, None] < r[None, :]`. The two occurrences are
// COM: the same SSA value but not the same element, so they must not cancel.
// COM: Cancelling them would give d = 0 and refute the comparison, which is
// COM: true for every lane above the diagonal. The axis placement keeps the
// COM: two symbols distinct, and the verdict is Unknown - the only sound
// COM: answer here, since neither Satisfied nor Refuted holds element-wise.

// CHECK-LABEL: tt.func @two_axes
module {
  tt.func @two_axes() {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %col = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %row = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %cb = tt.broadcast %col : tensor<64x1xi32> -> tensor<64x64xi32>
    %rb = tt.broadcast %row : tensor<1x64xi32> -> tensor<64x64xi32>
    // expected-remark@+1 {{verdict: Unknown}}
    %cmp = arith.cmpi slt, %cb, %rb : tensor<64x64xi32>
    tt.return
  }
}

// -----

// COM: A layout change moves no element, so `ttg.convert_layout`
// COM: is transparent to normalization and the lane symbol on either side of
// COM: it is the same symbol: `c < c + 1` is Satisfied element-wise. Were the
// COM: conversion opaque instead, the two sides would be unrelated symbols
// COM: and the verdict Unknown. This needs a TTGIR module with real layout
// COM: attributes, which is why it lives here rather than in the gtest.

#b1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#b2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: tt.func @convert
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @convert() {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #b1>
    %c = ttg.convert_layout %r : tensor<32xi32, #b1> -> tensor<32xi32, #b2>
    %one = arith.constant dense<1> : tensor<32xi32, #b2>
    %c1 = arith.addi %c, %one : tensor<32xi32, #b2>
    // expected-remark@+1 {{verdict: Satisfied}}
    %cmp = arith.cmpi slt, %c, %c1 : tensor<32xi32, #b2>
    tt.return
  }
}
