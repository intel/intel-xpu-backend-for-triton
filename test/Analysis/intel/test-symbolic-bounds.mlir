// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-symbolic-bounds -verify-diagnostics=only-expected | FileCheck %s

// COM: The prover's rules are unit-tested directly in
// COM: unittest/Analysis/SymbolicBoundsTest.cpp. The sections here are the
// COM: ones that need a whole module: the element-correspondence case that
// COM: must never be refuted, and a TTGIR layout conversion, which a gtest
// COM: string cannot carry.
// COM:
// COM: `-verify-diagnostics=only-expected` ignores unannotated remarks, so a
// COM: comparison with no `expected-remark` above it asserts nothing; only the
// COM: annotated ones are pinned. There is no `scf.for` remark: the prover
// COM: has no trip-count API.

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
