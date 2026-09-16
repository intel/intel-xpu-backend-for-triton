// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: env TRITON_INTEL_REDUCE_USE_COMMON_LOWERING=1 triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Verify that f32 within-thread reduction uses tree reduction (parallel pairs)
// COM: while f16 and integer types use left-fold (sequential accumulation) to
// COM: preserve low-precision accuracy. See issue #6904 and PR #6667.
// COM:
// COM: The second RUN line lowers through the common upstream pattern instead of
// COM: the Intel one (issue #6719). It shares this file's CHECK lines on purpose:
// COM: the common lowering takes its association solely from
// COM: TargetInfo::getReductionTreeArity, so passing under both RUN lines is what
// COM: demonstrates the Intel arity override reproduces the forked fold order.

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

// -----

// COM: Non-float types also left-fold, regardless of width: the predicate is
// COM: "float and >= 32 bits" for tree reduction, so i32 folds. This is the
// COM: majority of the left-folded cases, so it must be covered.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_i32_left_fold
  tt.func @reduce_i32_left_fold(%f : tensor<128xi32, #blocked>) -> i32 {
    // CHECK: llvm.extractvalue {{.*}}[0]
    // CHECK: llvm.extractvalue {{.*}}[1]
    // CHECK: llvm.extractvalue {{.*}}[2]
    // CHECK: llvm.extractvalue {{.*}}[3]
    // CHECK: [[S0:%.*]] = llvm.add %{{.*}}, %{{.*}} : i32
    // CHECK: [[S1:%.*]] = llvm.add [[S0]], %{{.*}} : i32
    // CHECK: llvm.add [[S1]], %{{.*}} : i32
    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<128xi32, #blocked>) -> i32
    tt.return %g : i32
  }
}

// -----

// COM: Multi-operand reduce, value operand first. The fold order is classified
// COM: from the *source* tensor's operand 0, which is f32 here, so this trees.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_argmax_value_first
  tt.func @reduce_argmax_value_first(%f : tensor<128xf32, #blocked>, %i : tensor<128xi32, #blocked>) -> (f32, i32) {
    // COM: CHECK-NEXT throughout: with plain CHECK, FileCheck can skip over the
    // COM: intermediate combine and satisfy a fold-order pattern under the other
    // COM: association, so a loose sequence here does not discriminate.
    // CHECK:      [[C0:%.*]] = llvm.fcmp "ogt" %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: [[V0:%.*]] = llvm.select [[C0]], %{{.*}}, %{{.*}} : i1, f32
    // CHECK-NEXT: [[I0:%.*]] = llvm.select [[C0]], %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: [[C1:%.*]] = llvm.fcmp "ogt" %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: [[V1:%.*]] = llvm.select [[C1]], %{{.*}}, %{{.*}} : i1, f32
    // CHECK-NEXT: [[I1:%.*]] = llvm.select [[C1]], %{{.*}}, %{{.*}} : i1, i32
    // COM: Tree: the third compare consumes the two partial results.
    // CHECK-NEXT: [[C2:%.*]] = llvm.fcmp "ogt" [[V0]], [[V1]] : f32
    // CHECK-NEXT: llvm.select [[C2]], [[V0]], [[V1]] : i1, f32
    // CHECK-NEXT: llvm.select [[C2]], [[I0]], [[I1]] : i1, i32
    %v, %idx = "tt.reduce" (%f, %i) ({
    ^bb0(%v0: f32, %i0: i32, %v1: f32, %i1: i32):
      %gt = arith.cmpf ogt, %v0, %v1 : f32
      %rv = arith.select %gt, %v0, %v1 : f32
      %ri = arith.select %gt, %i0, %i1 : i32
      tt.reduce.return %rv, %ri : f32, i32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> (f32, i32)
    tt.return %v, %idx : f32, i32
  }
}

// -----

// COM: The same reduction with the index operand first. Operand 0 is now i32, so
// COM: this left-folds. Together with the case above this is what distinguishes
// COM: "classify from source operand 0" from a rule based on the combiner's own
// COM: operand types: a single ordering cannot tell the two apart.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_argmax_index_first
  tt.func @reduce_argmax_index_first(%i : tensor<128xi32, #blocked>, %f : tensor<128xf32, #blocked>) -> (i32, f32) {
    // CHECK:      [[C0:%.*]] = llvm.fcmp "ogt" %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: [[I0:%.*]] = llvm.select [[C0]], %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: [[V0:%.*]] = llvm.select [[C0]], %{{.*}}, %{{.*}} : i1, f32
    // COM: Left fold: each compare consumes the running accumulator.
    // CHECK-NEXT: [[C1:%.*]] = llvm.fcmp "ogt" [[V0]], %{{.*}} : f32
    // CHECK-NEXT: [[I1:%.*]] = llvm.select [[C1]], [[I0]], %{{.*}} : i1, i32
    // CHECK-NEXT: [[V1:%.*]] = llvm.select [[C1]], [[V0]], %{{.*}} : i1, f32
    // CHECK-NEXT: [[C2:%.*]] = llvm.fcmp "ogt" [[V1]], %{{.*}} : f32
    // CHECK-NEXT: llvm.select [[C2]], [[I1]], %{{.*}} : i1, i32
    // CHECK-NEXT: llvm.select [[C2]], [[V1]], %{{.*}} : i1, f32
    %idx, %v = "tt.reduce" (%i, %f) ({
    ^bb0(%i0: i32, %v0: f32, %i1: i32, %v1: f32):
      %gt = arith.cmpf ogt, %v0, %v1 : f32
      %ri = arith.select %gt, %i0, %i1 : i32
      %rv = arith.select %gt, %v0, %v1 : f32
      tt.reduce.return %ri, %rv : i32, f32
    }) {axis = 0 : i32} : (tensor<128xi32, #blocked>, tensor<128xf32, #blocked>) -> (i32, f32)
    tt.return %idx, %v : i32, f32
  }
}

// -----

// COM: Sub-byte types left-fold like any other non-float type. This case also
// COM: pins the scratch sizing: the reduction is cross-warp, and both the legacy
// COM: shape-based allocator and getScratchSizeInBytes() round i1 up to a byte
// COM: (tt::getBitwidth clamps to 8), so neither path allocates zero.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: ttg.shared = 4 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: reduce_i1_cross_warp
  tt.func @reduce_i1_cross_warp(%f : tensor<512xi1, #blocked>) -> i1 {
    // CHECK: [[S0:%.*]] = llvm.or %{{.*}}, %{{.*}} : i1
    // CHECK: [[S1:%.*]] = llvm.or [[S0]], %{{.*}} : i1
    // CHECK: llvm.or [[S1]], %{{.*}} : i1
    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i1, %arg1: i1):
      %or = arith.ori %arg0, %arg1 : i1
      tt.reduce.return %or : i1
    }) {axis = 0 : i32} : (tensor<512xi1, #blocked>) -> i1
    tt.return %g : i1
  }
}
