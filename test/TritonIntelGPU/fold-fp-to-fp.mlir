// fast-math off (default): only chains with a lossless inner cast fold.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-fold-fp-to-fp | FileCheck %s --check-prefixes=CHECK,NOFM
// fast-math on: chains with a lossy inner cast also fold.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-fold-fp-to-fp=fast-math=true | FileCheck %s --check-prefixes=CHECK,FM

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: f32 -> f8E5M2 -> f16. Inner f32->f8 is a real (lossy) downcast; outer
// COM: f8->f16 is lossless. Folds only under fast-math.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8_f16
  tt.func @fold_f32_f8_f16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf16, #mma> {
    // FM: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // FM-NOT: f8E5M2
    // NOFM: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    // NOFM: tt.fp_to_fp %[[P]] : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    tt.return %1 : tensor<128x32xf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Lossy outer cast (f16 -> f8E5M2). Outer is not a widen, so the chain
// COM: never folds, regardless of fast-math.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_lossy_outer
  tt.func @no_fold_lossy_outer(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf8E5M2, #mma> {
    // CHECK: tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // CHECK: tt.fp_to_fp {{.*}} : tensor<128x32xf16, #mma> -> tensor<128x32xf8E5M2, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0, rounding = rtne : tensor<128x32xf16, #mma> -> tensor<128x32xf8E5M2, #mma>
    tt.return %1 : tensor<128x32xf8E5M2, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Differing-tradeoff intermediate. f8E4M3 is NOT a subset of f8E5M2 (more
// COM: mantissa, less exponent range), so outer cast is lossy -> never folds.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_f8e4m3_f8e5m2
  tt.func @no_fold_f8e4m3_f8e5m2(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf8E5M2, #mma> {
    // CHECK: tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    // CHECK: tt.fp_to_fp {{.*}} -> tensor<128x32xf8E5M2, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    %1 = tt.fp_to_fp %0, rounding = rtne : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xf8E5M2, #mma>
    tt.return %1 : tensor<128x32xf8E5M2, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Differing-tradeoff intermediate at 16 bits: bf16 is NOT a subset of f16
// COM: (more exponent range, less mantissa), so outer cast is lossy -> never folds.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_bf16_f16
  tt.func @no_fold_bf16_f16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf16, #mma> {
    // CHECK: tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xbf16, #mma>
    // CHECK: tt.fp_to_fp {{.*}} -> tensor<128x32xf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xbf16, #mma>
    %1 = tt.fp_to_fp %0, rounding = rtne : tensor<128x32xbf16, #mma> -> tensor<128x32xf16, #mma>
    tt.return %1 : tensor<128x32xf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Pure widen chain f8E5M2 -> f16 -> f32. Both inner and outer casts are
// COM: lossless, so dropping f16 preserves the result exactly. Always folds,
// COM: even with fast-math off (strict-FP legal). Merged op needs no rounding.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_widen_chain_no_rounding
  tt.func @fold_widen_chain_no_rounding(%arg0: tensor<128x32xf8E5M2, #mma>) -> tensor<128x32xf32, #mma> {
    // CHECK: %[[R:.*]] = tt.fp_to_fp %arg0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf32, #mma>
    // CHECK-NOT: f16
    %0 = tt.fp_to_fp %arg0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf16, #mma> -> tensor<128x32xf32, #mma>
    tt.return %1 : tensor<128x32xf32, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: f32 -> f16 -> f64. Inner f32->f16 is lossy (downcast); outer f16->f64
// COM: is lossless. Folds only under fast-math; merged f32 -> f64 is itself a
// COM: pure widen so it drops its rounding mode.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f16_f64
  tt.func @fold_f32_f16_f64(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf64, #mma> {
    // FM: %[[R:.*]] = tt.fp_to_fp %arg0 : tensor<128x32xf32, #mma> -> tensor<128x32xf64, #mma>
    // FM-NOT: f16
    // NOFM: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // NOFM: tt.fp_to_fp %[[P]] : tensor<128x32xf16, #mma> -> tensor<128x32xf64, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf16, #mma> -> tensor<128x32xf64, #mma>
    tt.return %1 : tensor<128x32xf64, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Mirrors the nvfp4 + swiglu epilogue failure (issue #6993). A user
// COM: downcast to f8E4M3FN feeds the f8 -> bf16 upcast that AccelerateMatmul
// COM: inserts to satisfy DPAS input limits. Inner f32->f8E4M3FN is a real
// COM: (lossy) downcast; outer f8E4M3FN->bf16 is a lossless widen. Folding to
// COM: f32 -> bf16 drops the f8 round-trip and is more accurate than the
// COM: reference, so it must fold only under fast-math.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8e4m3_bf16
  tt.func @fold_f32_f8e4m3_bf16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xbf16, #mma> {
    // FM: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xbf16, #mma>
    // FM-NOT: f8E4M3FN
    // NOFM: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    // NOFM: tt.fp_to_fp %[[P]] : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xbf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xbf16, #mma>
    tt.return %1 : tensor<128x32xbf16, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

// COM: Same nvfp4 + swiglu chain as above but on a blocked layout, as it
// COM: appears after remove-layout-conversions in the real pipeline. The fold
// COM: is decided from element types only, so the blocked encoding folds
// COM: identically: f32 -> f8E4M3FN -> bf16 collapses to f32 -> bf16 under
// COM: fast-math.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8e4m3_bf16_blocked
  tt.func @fold_f32_f8e4m3_bf16_blocked(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xbf16, #blocked> {
    // FM: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #blocked> -> tensor<128x32xbf16, #blocked>
    // FM-NOT: f8E4M3FN
    // NOFM: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E4M3FN, #blocked>
    // NOFM: tt.fp_to_fp %[[P]] : tensor<128x32xf8E4M3FN, #blocked> -> tensor<128x32xbf16, #blocked>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E4M3FN, #blocked>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E4M3FN, #blocked> -> tensor<128x32xbf16, #blocked>
    tt.return %1 : tensor<128x32xbf16, #blocked>
  }
}
