// RUN: triton-opt %s -split-input-file -tritonintelgpu-fold-fp-to-fp | FileCheck %s

// COM: Fast-math is opt-in via the `ttig.fast_math` module attribute (set from
// COM: TRITON_INTEL_FAST_MATH by the annotate-module pass). Modules carrying the
// COM: attribute allow the lossy fold; modules without it do not.

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: f32 -> f8E5M2 -> f16, no fast-math. Inner f32->f8 is a real (lossy)
// COM: downcast, so without fast-math the chain is kept intact.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_f32_f8_f16
  tt.func @no_fold_f32_f8_f16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf16, #mma> {
    // CHECK: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    // CHECK: tt.fp_to_fp %[[P]] : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    tt.return %1 : tensor<128x32xf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Same f32 -> f8E5M2 -> f16 chain, now with fast-math. The lossy f8
// COM: intermediate is dropped, leaving a single f32 -> f16 rtne.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8_f16
  tt.func @fold_f32_f8_f16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf16, #mma> {
    // CHECK: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // CHECK-NOT: f8E5M2
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    tt.return %1 : tensor<128x32xf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Lossy outer cast (f16 -> f8E5M2). Outer is not a widen, so the chain
// COM: never folds even with fast-math enabled.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
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
// COM: mantissa, less exponent range), so outer cast is lossy -> never folds,
// COM: even with fast-math.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
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
// COM: (more exponent range, less mantissa), so outer cast is lossy -> never
// COM: folds, even with fast-math.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
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
// COM: lossless, so dropping f16 preserves the result exactly. Folds even
// COM: WITHOUT fast-math (strict-FP legal). Merged op needs no rounding.
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

// COM: f32 -> f16 -> f64, no fast-math. Inner f32->f16 is a lossy downcast, so
// COM: without fast-math the chain is kept intact.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_f32_f16_f64
  tt.func @no_fold_f32_f16_f64(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf64, #mma> {
    // CHECK: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // CHECK: tt.fp_to_fp %[[P]] : tensor<128x32xf16, #mma> -> tensor<128x32xf64, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf16, #mma> -> tensor<128x32xf64, #mma>
    tt.return %1 : tensor<128x32xf64, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Same f32 -> f16 -> f64 chain, now with fast-math. The lossy f16 hop is
// COM: dropped; the merged f32 -> f64 is a pure widen so it drops its rounding.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f16_f64
  tt.func @fold_f32_f16_f64(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf64, #mma> {
    // CHECK: %[[R:.*]] = tt.fp_to_fp %arg0 : tensor<128x32xf32, #mma> -> tensor<128x32xf64, #mma>
    // CHECK-NOT: f16
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf16, #mma> -> tensor<128x32xf64, #mma>
    tt.return %1 : tensor<128x32xf64, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Mirrors the nvfp4 + swiglu epilogue failure (issue #6993), no fast-math.
// COM: A user downcast to f8E4M3FN feeds the f8 -> bf16 upcast AccelerateMatmul
// COM: inserts for DPAS. Inner f32->f8E4M3FN is a lossy downcast, so without
// COM: fast-math the chain is kept (matching the reference's f8 round-trip).
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_fold_f32_f8e4m3_bf16
  tt.func @no_fold_f32_f8e4m3_bf16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xbf16, #mma> {
    // CHECK: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    // CHECK: tt.fp_to_fp %[[P]] : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xbf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xbf16, #mma>
    tt.return %1 : tensor<128x32xbf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Same nvfp4 + swiglu chain, now with fast-math: the f8 round-trip is
// COM: dropped, collapsing to f32 -> bf16 (more accurate than the reference).
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8e4m3_bf16
  tt.func @fold_f32_f8e4m3_bf16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xbf16, #mma> {
    // CHECK: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xbf16, #mma>
    // CHECK-NOT: f8E4M3FN
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E4M3FN, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E4M3FN, #mma> -> tensor<128x32xbf16, #mma>
    tt.return %1 : tensor<128x32xbf16, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

// COM: Same nvfp4 + swiglu chain on a blocked layout (as it appears after
// COM: remove-layout-conversions), with fast-math. The fold is decided from
// COM: element types only, so the blocked encoding folds identically:
// COM: f32 -> f8E4M3FN -> bf16 collapses to f32 -> bf16.
module attributes {ttig.fast_math, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8e4m3_bf16_blocked
  tt.func @fold_f32_f8e4m3_bf16_blocked(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xbf16, #blocked> {
    // CHECK: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #blocked> -> tensor<128x32xbf16, #blocked>
    // CHECK-NOT: f8E4M3FN
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #blocked> -> tensor<128x32xf8E4M3FN, #blocked>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E4M3FN, #blocked> -> tensor<128x32xbf16, #blocked>
    tt.return %1 : tensor<128x32xbf16, #blocked>
  }
}
