// default (unset) -> fold ON
// RUN: triton-opt %s -split-input-file -tritonintelgpu-accelerate-matmul | FileCheck %s --check-prefixes=CHECK,FOLD
// explicit on -> fold
// RUN: env TRITON_INTEL_FOLD_LOSSY_FPCAST=1 triton-opt %s -split-input-file -tritonintelgpu-accelerate-matmul | FileCheck %s --check-prefixes=CHECK,FOLD
// explicit opt-out -> keep both casts
// RUN: env TRITON_INTEL_FOLD_LOSSY_FPCAST=0 triton-opt %s -split-input-file -tritonintelgpu-accelerate-matmul | FileCheck %s --check-prefixes=CHECK,NOFOLD

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: f32 -> f8E5M2 -> f16. Outer f8E5M2->f16 is lossless (f8E5M2 subset of f16),
// COM: so the narrow fp8 intermediate is dropped, leaving a single f32 -> f16 rtne.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_f32_f8_f16
  tt.func @fold_f32_f8_f16(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf16, #mma> {
    // FOLD: %[[R:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf16, #mma>
    // FOLD-NOT: f8E5M2
    // NOFOLD: %[[P:.*]] = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    // NOFOLD: tt.fp_to_fp %[[P]] : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x32xf32, #mma> -> tensor<128x32xf8E5M2, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    tt.return %1 : tensor<128x32xf16, #mma>
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: Lossy outer cast (f16 -> f8E5M2). Must NOT fold regardless of flag.
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
// COM: mantissa, less exponent range), so outer cast is lossy -> must NOT fold.
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

// COM: Value-preserving widen chain f8E5M2 -> f16 -> f32. Outer f16->f32 is
// COM: lossless, so it folds to a single f8E5M2 -> f32. The merged A -> C is a
// COM: pure widen, so no rounding mode is attached.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fold_widen_chain_no_rounding
  tt.func @fold_widen_chain_no_rounding(%arg0: tensor<128x32xf8E5M2, #mma>) -> tensor<128x32xf32, #mma> {
    // FOLD: %[[R:.*]] = tt.fp_to_fp %arg0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf32, #mma>
    // FOLD-NOT: f16
    // NOFOLD: %[[P:.*]] = tt.fp_to_fp %arg0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    // NOFOLD: tt.fp_to_fp %[[P]] : tensor<128x32xf16, #mma> -> tensor<128x32xf32, #mma>
    %0 = tt.fp_to_fp %arg0 : tensor<128x32xf8E5M2, #mma> -> tensor<128x32xf16, #mma>
    %1 = tt.fp_to_fp %0 : tensor<128x32xf16, #mma> -> tensor<128x32xf32, #mma>
    tt.return %1 : tensor<128x32xf32, #mma>
  }
}
