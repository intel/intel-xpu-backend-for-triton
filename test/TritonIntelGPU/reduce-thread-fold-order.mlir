// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Verify that f32 within-thread reduction uses tree reduction (parallel pairs)
// COM: while f16 uses left-fold (sequential accumulation) to preserve low-precision
// COM: accuracy. See issue #6904 and PR #6667.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_f32_tree
  tt.func @reduce_f32_tree(%f : tensor<128xf32, #blocked>) -> f32 {
    // 4 registers per thread, tree reduce: (r0+r1), (r2+r3), then combine
    // CHECK: llvm.extractvalue {{.*}}[0]
    // CHECK: llvm.extractvalue {{.*}}[1]
    // CHECK: llvm.extractvalue {{.*}}[2]
    // CHECK: llvm.extractvalue {{.*}}[3]
    // CHECK: [[A:%.*]] = llvm.fadd %{{.*}}, %{{.*}} : f32
    // CHECK: [[B:%.*]] = llvm.fadd %{{.*}}, %{{.*}} : f32
    // CHECK: llvm.fadd [[A]], [[B]] : f32
    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return %g : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_f16_left_fold
  tt.func @reduce_f16_left_fold(%f : tensor<128xf16, #blocked>) -> f16 {
    // 4 registers per thread, left fold: r0+r1 -> +r2 -> +r3 (sequential chain)
    // CHECK: llvm.extractvalue {{.*}}[0]
    // CHECK: llvm.extractvalue {{.*}}[1]
    // CHECK: llvm.extractvalue {{.*}}[2]
    // CHECK: llvm.extractvalue {{.*}}[3]
    // CHECK: [[S0:%.*]] = llvm.fadd %{{.*}}, %{{.*}} : f16
    // CHECK: [[S1:%.*]] = llvm.fadd [[S0]], %{{.*}} : f16
    // CHECK: llvm.fadd [[S1]], %{{.*}} : f16
    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: f16, %arg1: f16):
      %add = arith.addf %arg0, %arg1 : f16
      tt.reduce.return %add : f16
    }) {axis = 0 : i32} : (tensor<128xf16, #blocked>) -> f16
    tt.return %g : f16
  }
}
