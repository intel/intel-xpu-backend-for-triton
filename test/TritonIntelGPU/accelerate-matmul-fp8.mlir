// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_non_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    // COM: Checks that we convert to fp16 when support_fp8_dpas is not set
    // CHECK: %[[CONV1:.*]] = ttg.convert_layout %arg0 : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf8E5M2
    // CHECK: %[[CONV2:.*]] = ttg.convert_layout %arg1 : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf8E5M2
    // CHECK: %[[FPTOFP1:.*]] = tt.fp_to_fp %[[CONV1]] : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf16
    // CHECK: %[[FPTOFP2:.*]] = tt.fp_to_fp %[[CONV2]] : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf16
    // CHECK: tt.dot %[[FPTOFP1]], %[[FPTOFP2]]{{.*}} : tensor<128x128xf16{{.*}}*{{.*}}tensor<128x128xf16{{.*}}-> tensor<128x128xf32

    %res = tt.dot %arg1, %arg2, %cst : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %52 = ttg.convert_layout %res : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    %53 = arith.extf %52 : tensor<128x128xf32, #blocked1> to tensor<128x128xf64, #blocked1>
    tt.store %arg3, %53 : tensor<128x128x!tt.ptr<f64>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_non_k_pack_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x64xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x4xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<128x64xi8, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_non_k_pack_dot_scaled(%a: tensor<128x64xi8, #blocked2>, %scale_a: tensor<128x4xi8, #blocked1>, %b: tensor<128x64xi8, #blocked>, %scale_b: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // COM: does not support non-k packed matrix. Decompose the tt.dot_scaled.
    // CHECK-NOT: tt.dot_scaled
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, rhs_k_pack = false} : tensor<128x64xi8, #blocked2>, tensor<128x4xi8, #blocked1> * tensor<128x64xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_non_k_pack_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<64x128xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x4xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xi8, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_non_k_pack_dot_scaled(%a: tensor<64x128xi8, #blocked2>, %scale_a: tensor<128x4xi8, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // COM: does not support non-k packed matrix. Decompose the tt.dot_scaled.
    // CHECK-NOT: tt.dot_scaled
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false} : tensor<64x128xi8, #blocked2>, tensor<128x4xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
