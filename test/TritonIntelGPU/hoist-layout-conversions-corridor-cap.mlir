// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=1" | FileCheck %s --check-prefixes=CHECK,CAP1
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=2" | FileCheck %s --check-prefixes=CHECK,CAP2
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=1" 2>&1 | FileCheck %s --check-prefix=STATS1
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=2" 2>&1 | FileCheck %s --check-prefix=STATS2

// COM: The corridor operation cap bounds what the whole-function peak projection
// COM: will price. The projection walks every operation between a hoist's
// COM: insertion point and the conversion's old position, and each one costs a
// COM: fresh liveness set; the cap is what stops a pathological input from
// COM: turning that walk into the pass's dominant cost. Past the cap the walk is
// COM: abandoned and every unwalked point is charged conservatively as
// COM: prePeak + dstBytes.
// COM:
// COM: The default cap is far above anything a real kernel reaches, so it can only
// COM: be exercised by lowering it. This file lowers it around a fixed pair of
// COM: hoists whose corridors are 2 and 3 operations long, which is why the same
// COM: IR gives two different outcomes:
// COM:   cap=1: both corridors exceed the cap, both hoists are charged
// COM:          prePeak + 64 = 992 > 928 and both are refused.
// COM:   cap=2: the 2-operation corridor is priced (608, accepted at the 928
// COM:          ceiling) and only the 3-operation one bails.
// COM: Both hoists are free -- each retires a 256-byte source in exchange for a
// COM: 64-byte result -- so the loop-level gate passes them in every
// COM: configuration and the verdicts here are the corridor cap's alone. This is
// COM: the cap's cost made visible: it is not a speed knob, it buys bounded work
// COM: with lost hoists.
// COM:
// COM: The two loops are decided last-to-first (see the sibling-loop case in
// COM: hoist-layout-conversions.mlir), which at cap=2 is why the *second* loop's
// COM: conversion is the one that moves: it is decided first, and its corridor is
// COM: shorter by the loop the other candidate has to step over.
// COM:
// COM: A bail also resets the priced-operation count to 0 in the debug output
// COM: ("corridor=992 over 0 ops"), which distinguishes a conservative charge from
// COM: a walk that priced operations and found a genuine peak.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @corridor_cap_two_free_hoists
  tt.func @corridor_cap_two_free_hoists(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<128x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %s1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    %s2 = arith.addf %arg1, %arg1 : tensor<128x16xf16, #blocked>

    // COM: cap=1: neither conversion moves, and both are stamped tt.no_licm so
    // COM: that LICM does not undo the refusal.
    // CAP1: arith.addf
    // CAP1-NEXT: arith.addf
    // CAP1-NEXT: scf.for
    // CAP1-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP1: scf.for
    // CAP1-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>

    // COM: cap=2: the second loop's conversion moves up beside %s2, the first
    // COM: loop's stays put.
    // CAP2: arith.addf
    // CAP2-NEXT: arith.addf
    // CAP2-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP2-NEXT: scf.for
    // CAP2-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP2: scf.for
    // CAP2-NOT: ttg.convert_layout
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt1 = ttg.convert_layout %s1 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %d1 = tt.dot %cvt1, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d1 : tensor<128x16xf32, #dpas>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %r1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt2 = ttg.convert_layout %s2 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %d2 = tt.dot %cvt2, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d2 : tensor<128x16xf32, #dpas>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas>
  }
}

// COM: Both refusals are counted as *fallback* rejections rather than exact ones:
// COM: the projection that produced them never walked the corridor.
// STATS1: [HoistLayoutConversions] considered=2 hoisted=0 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=2 skipped_other=0
// STATS2: [HoistLayoutConversions] considered=2 hoisted=1 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=1 skipped_other=0
