// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// COM: The block scaled DPAS takes E8M0 scales, one per 32 K elements: the SPIR-V extension it lowers to
// COM: (SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate) has no other scale format, and the lowering
// COM: packs each scale byte into the DPAS operand as an E8M0 exponent. A tt.dot_scaled with any other
// COM: scale is decomposed instead (issue #8881).

// COM: NVFP4: e2m1 x e2m1 with E4M3 scales, one per 16 K elements (K = 64, 4 scales per row).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @nvfp4_e4m3_scales_are_decomposed(
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.dot {{.*}} -> tensor<32x32xf32, #[[$DPAS:.+]]>
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.return
  tt.func public @nvfp4_e4m3_scales_are_decomposed(%a: tensor<32x32xi8, #blocked>, %scale_a: tensor<32x4xf8E4M3FN, #blocked1>, %b: tensor<32x32xi8, #blocked>, %scale_b: tensor<32x4xf8E4M3FN, #blocked1>) -> tensor<32x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x32xi8, #blocked>, tensor<32x4xf8E4M3FN, #blocked1> * tensor<32x32xi8, #blocked>, tensor<32x4xf8E4M3FN, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// COM: E8M0 scales, but one per 16 K elements: decomposed too.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @e8m0_scales_per_16_elements_are_decomposed(
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.dot {{.*}} -> tensor<32x32xf32, #[[$DPAS:.+]]>
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.return
  tt.func public @e8m0_scales_per_16_elements_are_decomposed(%a: tensor<32x32xi8, #blocked>, %scale_a: tensor<32x4xi8, #blocked1>, %b: tensor<32x32xi8, #blocked>, %scale_b: tensor<32x4xi8, #blocked1>) -> tensor<32x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x32xi8, #blocked>, tensor<32x4xi8, #blocked1> * tensor<32x32xi8, #blocked>, tensor<32x4xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// COM: E4M3 scales, one per 32 K elements: decomposed, as the scale format is not E8M0.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @e4m3_scales_per_32_elements_are_decomposed(
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.dot {{.*}} -> tensor<32x32xf32, #[[$DPAS:.+]]>
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.return
  tt.func public @e4m3_scales_per_32_elements_are_decomposed(%a: tensor<32x64xf8E4M3FN, #blocked>, %scale_a: tensor<32x2xf8E4M3FN, #blocked1>, %b: tensor<64x32xf8E4M3FN, #blocked>, %scale_b: tensor<32x2xf8E4M3FN, #blocked1>) -> tensor<32x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x64xf8E4M3FN, #blocked>, tensor<32x2xf8E4M3FN, #blocked1> * tensor<64x32xf8E4M3FN, #blocked>, tensor<32x2xf8E4M3FN, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// COM: An e2m1 B operand packed along N is upcast for the block scaled DPAS when its scales are E8M0
// COM: (see accelerate-matmul-fp8.mlir). With E4M3 scales it is decomposed instead, and the upcast does
// COM: not apply either: it would rebuild the same tt.dot_scaled with the same scales forever.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @nvfp4_packed_along_n_is_decomposed(
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.dot {{.*}} -> tensor<32x32xf32, #[[$DPAS:.+]]>
  // CHECK-NOT:     tt.dot_scaled
  // CHECK:         tt.return
  tt.func public @nvfp4_packed_along_n_is_decomposed(%a: tensor<32x32xi8, #blocked>, %scale_a: tensor<32x4xf8E4M3FN, #blocked1>, %b: tensor<64x16xi8, #blocked>, %scale_b: tensor<32x4xf8E4M3FN, #blocked1>) -> tensor<32x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, rhs_k_pack = false} : tensor<32x32xi8, #blocked>, tensor<32x4xf8E4M3FN, #blocked1> * tensor<64x16xi8, #blocked>, tensor<32x4xf8E4M3FN, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// COM: E8M0 scales, one per 32 K elements: the block scaled DPAS is used.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @mxfp4_e8m0_scales_use_block_scaled_dpas(
  // CHECK:         tt.dot_scaled {{.*}} -> tensor<32x32xf32, #[[$DPAS:.+]]>
  tt.func public @mxfp4_e8m0_scales_use_block_scaled_dpas(%a: tensor<32x32xi8, #blocked>, %scale_a: tensor<32x2xi8, #blocked1>, %b: tensor<32x32xi8, #blocked>, %scale_b: tensor<32x2xi8, #blocked1>) -> tensor<32x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x32xi8, #blocked>, tensor<32x2xi8, #blocked1> * tensor<32x32xi8, #blocked>, tensor<32x2xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}
