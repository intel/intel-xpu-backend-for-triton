// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// Test without support_subgroup_matrix_multiply_accumulate attribute (should NOT use DPAS)
// CHECK-NOT: ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_no_dpas_support
  tt.func public @kernel_no_dpas_support(
    %arg0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Without support attribute, should not use DPAS
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test mixed per-operation shape selection in one function with
// DPAS-compatible operand types.
// CHECK: #[[$SHAPE_DPAS:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_mixed_dpas_fma
  tt.func public @kernel_mixed_dpas_fma(
    %dpas_a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %dpas_b: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %fma_a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %fma_b: tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %dpas_out: tensor<16x16x!tt.ptr<f32>, #blocked>,
    %fma_out: tensor<16x8x!tt.ptr<f32>, #blocked>) {
    %dpas_acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %fma_acc = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[$SHAPE_DPAS]]>
    %dpas_result = tt.dot %dpas_a, %dpas_b, %dpas_acc : tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    // CHECK: ttg.convert_layout {{.*}} : tensor<16x16xf32, #[[$SHAPE_DPAS]]> -> tensor<16x16xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x8xf32, #blocked>
    %fma_result = tt.dot %fma_a, %fma_b, %fma_acc : tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x8xf32, #blocked>
    tt.store %dpas_out, %dpas_result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.store %fma_out, %fma_result : tensor<16x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test that a BF16 dot with K below the native DPAS K=16 remains blocked.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_small_k_fma
  tt.func public @kernel_small_k_fma(
    %a: tensor<16x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<8x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %out: tensor<16x16x!tt.ptr<f32>, #blocked>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #blocked>
    %result = tt.dot %a, %b, %acc : tensor<16x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.store %out, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test multiple dot operations in a function - all must be DPAS-compatible
// CHECK: #[[$DPAS:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_multiple_dots
  tt.func public @kernel_multiple_dots(
    %arg0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg3: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>,
    %arg5: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #[[$DPAS]]>
    %result1 = tt.dot %arg0, %arg1, %cst_f32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #[[$DPAS]]>
    %result2 = tt.dot %arg2, %arg3, %cst_f32 : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %result1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.store %arg5, %result2 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
// Tensor arguments lack a proven descriptor origin, including in forced modes.
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
// CHECK: ttig.gemm_feature_vectors = []
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @prefetched_dot_cluster
  tt.func public @prefetched_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 512 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 1 : i32} {
  // CHECK-LABEL: tt.func public @one_stage_dot_cluster
  tt.func public @one_stage_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 512 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 3 : i32} {
  // CHECK-LABEL: tt.func public @local_one_stage_dot_cluster
  tt.func public @local_one_stage_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 512 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    } {tt.num_stages = 1 : i32}
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-LABEL: tt.func public @legacy_target_dot_cluster
  tt.func public @legacy_target_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 512 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}
// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @rectangular_dot_cluster
  tt.func public @rectangular_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x128xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x128xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 512 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x128xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x128xf32, #blocked>
      scf.yield %dot : tensor<256x128xf32, #blocked>
    }
    tt.return %result : tensor<256x128xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @short_reduction_dot_cluster
  tt.func public @short_reduction_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 8192 : index
    %step = arith.constant 32 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// Tensor arguments lack a proven descriptor origin, including in forced modes.
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
// CHECK: ttig.gemm_feature_vectors = []
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @long_boundary_dot_cluster
  tt.func public @long_boundary_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 8193 : index
    %step = arith.constant 32 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 1 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @dynamic_reduction_dot_cluster
  tt.func public @dynamic_reduction_dot_cluster(%hi: index, %a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 4 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @dynamic_intervention_dot_cluster
  tt.func public @dynamic_intervention_dot_cluster(%hi: index, %a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %step = arith.constant 1 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}

// -----
// Tensor arguments lack a proven descriptor origin, including in forced modes.
// CHECK: #{{.*}} = #ttig.dpas<{{.*}}repCluster = [4, 2]
// CHECK: ttig.gemm_feature_vectors = []
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.gemm_policy_mode = 4 : i32, ttig.dpas_prefetch_stages = 2 : i32} {
  // CHECK-LABEL: tt.func public @short_intervention_dot_cluster
  tt.func public @short_intervention_dot_cluster(%a: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %lo = arith.constant 0 : index
    %hi = arith.constant 8192 : index
    %step = arith.constant 32 : index
    %result = scf.for %iv = %lo to %hi step %step iter_args(%acc = %zero) -> (tensor<256x256xf32, #blocked>) {
      %dot = tt.dot %a, %b, %acc : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %dot : tensor<256x256xf32, #blocked>
    }
    tt.return %result : tensor<256x256xf32, #blocked>
  }
}
