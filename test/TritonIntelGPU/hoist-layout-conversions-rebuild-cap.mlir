// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions | FileCheck %s --check-prefixes=CHECK,UNCAPPED
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="peak-rebuild-cap=1" | FileCheck %s --check-prefixes=CHECK,CAP1
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions 2>&1 | FileCheck %s --check-prefix=STATS0
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="peak-rebuild-cap=1" 2>&1 | FileCheck %s --check-prefix=STATS1

// COM: The analysis rebuild cap bounds how many times one function may rebuild the
// COM: whole-function pressure analysis. Every accepted hoist invalidates it, and
// COM: the next candidate that needs a projection has to pay for a fresh one --
// COM: the analysis cannot be reused, because a projection made against a stale
// COM: prePeak bounds nothing. The cap is what keeps a function with very many
// COM: candidates from paying that quadratically.
// COM:
// COM: Reaching the cap is a *rejection*, not a fallback to a cheaper estimate.
// COM: There is no cheaper estimate available: without a current analysis the pass
// COM: has no prePeak to compare against, so it cannot bound the hoist at all and
// COM: refuses it. The counter it bumps is the fallback one, since the refusal is
// COM: an admission of missing information rather than a measured cost.
// COM:
// COM: Three sibling loops sharing nothing, each with its own free hoist (a
// COM: 256-byte source retired for a 64-byte result), so the loop-level gate
// COM: passes all three in every configuration and the difference between the two
// COM: runs below is the cap's alone. The first build is free -- the pass makes it
// COM: on demand and no hoist has happened yet -- so cap=1 pays for exactly one
// COM: rebuild and the third candidate it reaches is refused:
// COM:   loop 3 decided first, priced against the free initial build -> hoisted
// COM:   loop 2 decided next, spends the one permitted rebuild        -> hoisted
// COM:   loop 1 decided last, would need a second rebuild             -> refused
// COM: The order is last-to-first because a candidate's source credit depends on
// COM: its siblings' conversions having already moved out from under it.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @rebuild_cap_three_free_hoists
  tt.func @rebuild_cap_three_free_hoists(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<128x16xf16, #blocked>, %arg2: tensor<128x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %s1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    %s2 = arith.addf %arg1, %arg1 : tensor<128x16xf16, #blocked>
    %s3 = arith.addf %arg2, %arg2 : tensor<128x16xf16, #blocked>

    // COM: Uncapped: all three move, each landing beside its own source.
    // UNCAPPED: arith.addf
    // UNCAPPED-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // UNCAPPED-NEXT: arith.addf
    // UNCAPPED-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // UNCAPPED-NEXT: arith.addf
    // UNCAPPED-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // UNCAPPED-NEXT: scf.for
    // UNCAPPED-NOT: ttg.convert_layout

    // COM: cap=1: only the two later loops move; the first loop's conversion stays
    // COM: put, stamped tt.no_licm so LICM does not undo the refusal.
    // CAP1: arith.addf
    // CAP1-NEXT: arith.addf
    // CAP1-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP1-NEXT: arith.addf
    // CAP1-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP1-NEXT: scf.for
    // CAP1-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP1: scf.for
    // CAP1-NOT: ttg.convert_layout
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
    %r3 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %r2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt3 = ttg.convert_layout %s3 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %d3 = tt.dot %cvt3, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d3 : tensor<128x16xf32, #dpas>
    }
    tt.return %r3 : tensor<128x16xf32, #dpas>
  }
}

// STATS0: [HoistLayoutConversions] considered=3 hoisted=3 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=0 skipped_other=0
// STATS1: [HoistLayoutConversions] considered=3 hoisted=2 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=1 skipped_other=0
