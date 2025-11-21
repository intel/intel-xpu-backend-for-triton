// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // COM: Checks that we don't convert to fp16 when support_fp8_dpas is set
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

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
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

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
// CHECK: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK: #[[$LINEARLAYOUT1:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [8, 0], [16, 0], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
// CHECK: #[[$LINEARLAYOUT2:.+]] = #ttg.linear<{register = {{\[\[}}0, 1], [16, 0], [0, 2]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, fp4KPack = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
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
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_hf8_bdpas(
  // CHECK-SAME:      %[[A:.*]]: tensor<128x64xf8E4M3FN, #blocked>,
  // CHECK-SAME:      %[[SCALE_A:.*]]: tensor<128x2xi8, #blocked1>,
  // CHECK-SAME:      %[[B:.*]]: tensor<64x128xf8E5M2, #blocked2>,
  // CHECK-SAME:      %[[SCALE_B:.*]]: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked2> {
  tt.func public @kernel_hf8_bdpas(%a: tensor<128x64xf8E4M3FN, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E5M2, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
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
