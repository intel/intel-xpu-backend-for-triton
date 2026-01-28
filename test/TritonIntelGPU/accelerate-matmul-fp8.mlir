// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // COM: Checks that we don't convert to fp16 when 'support_subgroup_matrix_multiply_accumulate_bf8' is set
    // CHECK: tt.dot{{.*}} : tensor<128x128xf8E5M2{{.*}}*{{.*}}tensor<128x128xf8E5M2{{.*}}-> tensor<128x128xf32
    %res = tt.dot %arg1, %arg2, %cst : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %52 = ttg.convert_layout %res : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    %53 = arith.extf %52 : tensor<128x128xf32, #blocked1> to tensor<128x128xf64, #blocked1>
    tt.store %arg3, %53 : tensor<128x128x!tt.ptr<f64>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_non_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    // COM: Checks that we convert to fp16 when 'support_subgroup_matrix_multiply_accumulate_bf8' is not set.
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

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [8, 0], [16, 0], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [16, 0], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, fp4KPack = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_fp4_bdpas(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x64xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x4xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xi8, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_fp4_bdpas(%a: tensor<128x64xi8, #blocked2>, %scale_a: tensor<128x4xi8, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED2]]>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf32, #[[$BLOCKED2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[VAL_2:.*]] = ttg.convert_layout %[[A]] : tensor<128x64xi8, #[[$BLOCKED]]> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[VAL_3:.*]] = ttg.convert_layout %[[B]] : tensor<64x128xi8, #[[$BLOCKED2]]> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 4}>>
    // CHECK: %[[VAL_4:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x4xi8, #[[$BLOCKED1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT1]]>
    // CHECK: %[[VAL_5:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x4xi8, #[[$BLOCKED1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT2]]>
    // CHECK: tt.dot_scaled %[[VAL_2]] scale %[[VAL_4]], %[[VAL_3]] scale %[[VAL_5]], %[[VAL_1]] lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x4xi8, #[[$LINEARLAYOUT1]]> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 4}>>, tensor<128x4xi8, #[[$LINEARLAYOUT2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked2>, tensor<128x4xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}


// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_mixed_fp8_bdpas(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x64xf8E4M3FN, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x2xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xf8E5M2, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_mixed_fp8_bdpas(%a: tensor<128x64xf8E4M3FN, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E5M2, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED2]]>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf32, #[[$BLOCKED2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[VAL_2:.*]] = ttg.convert_layout %[[A]] : tensor<128x64xf8E4M3FN, #[[$BLOCKED]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[VAL_3:.*]] = ttg.convert_layout %[[B]] : tensor<64x128xf8E5M2, #[[$BLOCKED2]]> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 4}>>
    // CHECK: %[[VAL_4:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT1]]>
    // CHECK: %[[VAL_5:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT2]]>
    // CHECK: tt.dot_scaled %[[VAL_2]] scale %[[VAL_4]], %[[VAL_3]] scale %[[VAL_5]], %[[VAL_1]] lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x2xi8, #[[$LINEARLAYOUT1]]> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 4}>>, tensor<128x2xi8, #[[$LINEARLAYOUT2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E4M3FN, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E5M2, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
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
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
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

// -----

// CHECK: #[[$BLOCKED_0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED_1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED_2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$BLOCKED_3:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT_0:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = {{\[\[}}0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[\[}}0, 32], [0, 64], [0, 128], [1, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT_1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT_2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_non_k_pack_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x64xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x4xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<128x64xi8, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_non_k_pack_dot_scaled(%a: tensor<128x64xi8, #blocked2>, %scale_a: tensor<128x4xi8, #blocked1>, %b: tensor<128x64xi8, #blocked>, %scale_b: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[CONSTANT_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED_2]]>
    // CHECK: %[[VAL_0:.*]] = ttg.fp4_to_fp %[[A]] {axis = 1 : i32} : tensor<128x64xi8, #[[$BLOCKED_0]]> -> tensor<128x128xf16, #[[$BLOCKED_3]]>
    // CHECK: %[[VAL_1:.*]] = ttg.fp4_to_fp %[[B]] {axis = 1 : i32} : tensor<128x64xi8, #[[$BLOCKED_2]]> -> tensor<128x128xf16, #[[$LINEARLAYOUT_0]]>
    // CHECK: %[[CONVERT_LAYOUT_0:.*]] = ttg.convert_layout %[[CONSTANT_0]] : tensor<128x128xf32, #[[$BLOCKED_2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[CONVERT_LAYOUT_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf16, #[[$BLOCKED_3]]> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK: %[[CONVERT_LAYOUT_2:.*]] = ttg.convert_layout %[[VAL_1]] : tensor<128x128xf16, #[[$LINEARLAYOUT_0]]> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[CONVERT_LAYOUT_3:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x4xi8, #[[$BLOCKED_1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT_1]]>
    // CHECK: %[[CONVERT_LAYOUT_4:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x4xi8, #[[$BLOCKED_1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT_2]]>
    // CHECK: tt.dot_scaled %[[CONVERT_LAYOUT_1]] scale %[[CONVERT_LAYOUT_3]], %[[CONVERT_LAYOUT_2]] scale %[[CONVERT_LAYOUT_4]], %[[CONVERT_LAYOUT_0]] lhs = fp16 rhs = fp16 {fastMath = false} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>, tensor<128x4xi8, #[[$LINEARLAYOUT_1]]> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x4xi8, #[[$LINEARLAYOUT_2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, rhs_k_pack = false} : tensor<128x64xi8, #blocked2>, tensor<128x4xi8, #blocked1> * tensor<128x64xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// CHECK: #[[$BLOCKED_0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED_1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED_2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$BLOCKED_3:.+]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT_0:.+]] = #ttg.linear<{register = {{\[\[}}1, 0], [0, 64], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0]], lane = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 8]], warp = {{\[\[}}0, 16], [0, 32], [2, 0], [4, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT_1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT_2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_non_k_pack_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<64x128xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x4xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xi8, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_non_k_pack_dot_scaled(%a: tensor<64x128xi8, #blocked2>, %scale_a: tensor<128x4xi8, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b: tensor<128x4xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[CONSTANT_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED_2]]>
    // CHECK: %[[VAL_0:.*]] = ttg.fp4_to_fp %[[A]] {axis = 0 : i32} : tensor<64x128xi8, #[[$BLOCKED_0]]> -> tensor<128x128xf16, #[[$LINEARLAYOUT_0]]>
    // CHECK: %[[VAL_1:.*]] = ttg.fp4_to_fp %[[B]] {axis = 0 : i32} : tensor<64x128xi8, #[[$BLOCKED_2]]> -> tensor<128x128xf16, #[[$BLOCKED_3]]>
    // CHECK: %[[CONVERT_LAYOUT_0:.*]] = ttg.convert_layout %[[CONSTANT_0]] : tensor<128x128xf32, #[[$BLOCKED_2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[CONVERT_LAYOUT_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf16, #[[$LINEARLAYOUT_0]]> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK: %[[CONVERT_LAYOUT_2:.*]] = ttg.convert_layout %[[VAL_1]] : tensor<128x128xf16, #[[$BLOCKED_3]]> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[CONVERT_LAYOUT_3:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x4xi8, #[[$BLOCKED_1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT_1]]>
    // CHECK: %[[CONVERT_LAYOUT_4:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x4xi8, #[[$BLOCKED_1]]> -> tensor<128x4xi8, #[[$LINEARLAYOUT_2]]>
    // CHECK: tt.dot_scaled %[[CONVERT_LAYOUT_1]] scale %[[CONVERT_LAYOUT_3]], %[[CONVERT_LAYOUT_2]] scale %[[CONVERT_LAYOUT_4]], %[[CONVERT_LAYOUT_0]] lhs = fp16 rhs = fp16 {fastMath = false} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>, tensor<128x4xi8, #[[$LINEARLAYOUT_1]]> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x4xi8, #[[$LINEARLAYOUT_2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false} : tensor<64x128xi8, #blocked2>, tensor<128x4xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT0:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = {{\[\[}}0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[\[}}0, 32], [0, 64], [1, 0], [2, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_mixed_prec_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x32xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x2xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xf8E5M2, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_mixed_prec_dot_scaled(%a: tensor<128x32xi8, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E5M2, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED2]]>
    // CHECK: %[[A_BF16:.*]] = ttg.fp4_to_fp %[[A]] {axis = 1 : i32} : tensor<128x32xi8, #[[$BLOCKED]]> -> tensor<128x64xbf16, #[[$LINEARLAYOUT0]]>
    // CHECK: %[[B_BF16:.*]] = tt.fp_to_fp %[[B]] : tensor<64x128xf8E5M2, #[[$BLOCKED2]]> -> tensor<64x128xbf16, #[[$BLOCKED2]]>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf32, #[[$BLOCKED2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[VAL_2:.*]] = ttg.convert_layout %[[A_BF16]] : tensor<128x64xbf16, #[[$LINEARLAYOUT0]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK: %[[VAL_3:.*]] = ttg.convert_layout %[[B_BF16]] : tensor<64x128xbf16, #[[$BLOCKED2]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[VAL_4:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT1]]>
    // CHECK: %[[VAL_5:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT2]]>
    // CHECK: tt.dot_scaled %[[VAL_2]] scale %[[VAL_4]], %[[VAL_3]] scale %[[VAL_5]], %[[VAL_1]] lhs = bf16 rhs = bf16 {fastMath = false} : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>, tensor<128x2xi8, #[[$LINEARLAYOUT1]]> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x2xi8, #[[$LINEARLAYOUT2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E5M2, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT0:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = {{\[\[}}0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[\[}}0, 32], [0, 64], [1, 0], [2, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_mixed_fp4_fp8_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x32xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x2xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xf8E5M2, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_mixed_fp4_fp8_dot_scaled(%a: tensor<128x32xi8, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E5M2, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED2]]>
    // CHECK: %[[A_BF16:.*]] = ttg.fp4_to_fp %[[A]] {axis = 1 : i32} : tensor<128x32xi8, #[[$BLOCKED]]> -> tensor<128x64xbf16, #[[$LINEARLAYOUT0]]>
    // CHECK: %[[B_BF16:.*]] = tt.fp_to_fp %[[B]] : tensor<64x128xf8E5M2, #[[$BLOCKED2]]> -> tensor<64x128xbf16, #[[$BLOCKED2]]>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf32, #[[$BLOCKED2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[VAL_2:.*]] = ttg.convert_layout %[[A_BF16]] : tensor<128x64xbf16, #[[$LINEARLAYOUT0]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK: %[[VAL_3:.*]] = ttg.convert_layout %[[B_BF16]] : tensor<64x128xbf16, #[[$BLOCKED2]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[VAL_4:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT1]]>
    // CHECK: %[[VAL_5:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT2]]>
    // CHECK: tt.dot_scaled %[[VAL_2]] scale %[[VAL_4]], %[[VAL_3]] scale %[[VAL_5]], %[[VAL_1]] lhs = bf16 rhs = bf16 {fastMath = false} : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>, tensor<128x2xi8, #[[$LINEARLAYOUT1]]> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x2xi8, #[[$LINEARLAYOUT2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E5M2, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT0:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = {{\[\[}}0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[\[}}0, 32], [0, 64], [1, 0], [2, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_mixed_fp4_hf16_dot_scaled(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x32xi8, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x2xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xf16, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_mixed_fp4_hf16_dot_scaled(%a: tensor<128x32xi8, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf16, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #[[$BLOCKED2]]>
    // CHECK: %[[A_FP16:.*]] = ttg.fp4_to_fp %[[A]] {axis = 1 : i32} : tensor<128x32xi8, #[[$BLOCKED]]> -> tensor<128x64xf16, #[[$LINEARLAYOUT0]]>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout %[[VAL_0]] : tensor<128x128xf32, #[[$BLOCKED2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    // CHECK: %[[VAL_2:.*]] = ttg.convert_layout %[[A_FP16]] : tensor<128x64xf16, #[[$LINEARLAYOUT0]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK: %[[VAL_3:.*]] = ttg.convert_layout %[[B]] : tensor<64x128xf16, #[[$BLOCKED2]]> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK: %[[VAL_4:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT1]]>
    // CHECK: %[[VAL_5:.*]] = ttg.convert_layout %[[SCALE_B]] : tensor<128x2xi8, #[[$BLOCKED1]]> -> tensor<128x2xi8, #[[$LINEARLAYOUT2]]>
    // CHECK: tt.dot_scaled %[[VAL_2]] scale %[[VAL_4]], %[[VAL_3]] scale %[[VAL_5]], %[[VAL_1]] lhs = fp16 rhs = fp16 {fastMath = false} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>, tensor<128x2xi8, #[[$LINEARLAYOUT1]]> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>, tensor<128x2xi8, #[[$LINEARLAYOUT2]]> -> tensor<128x128xf32, #[[$DPAS]]>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = fp16 {fastMath = false} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf16, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
