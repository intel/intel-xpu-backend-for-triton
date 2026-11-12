// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// Test FP8 -> BF16 accumulation without FP8 DPAS (FP8 extended to FP16, FP32 accumulation, one RTNE rounding to BF16)
// CHECK: #[[$DPAS_FP8_BF16:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_fp8_to_bf16_dpas
  tt.func public @kernel_fp8_to_bf16_dpas(
    %arg0: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<128x64xf8E5M2, {{.*}}> -> tensor<128x64xf16, {{.*}}>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<64x128xf8E5M2, {{.*}}> -> tensor<64x128xf16, {{.*}}>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<128x128xbf16, #[[$DPAS_FP8_BF16]]> -> tensor<128x128xf32, #[[$DPAS_FP8_BF16]]>
    // CHECK: %[[D:.+]] = tt.dot {{.*}} : tensor<128x64xf16{{.*}}> * tensor<64x128xf16{{.*}}> -> tensor<128x128xf32, #[[$DPAS_FP8_BF16]]>
    // CHECK: tt.fp_to_fp %[[D]], rounding = rtne : tensor<128x128xf32, #[[$DPAS_FP8_BF16]]> -> tensor<128x128xbf16, #[[$DPAS_FP8_BF16]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Test mixed FP8 -> BF16 accumulation without FP8 DPAS (both extended to FP16, FP32 accumulation)
// CHECK: #[[$DPAS_MIXED_FP8_BF16:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_mixed_fp8_to_bf16_dpas
  tt.func public @kernel_mixed_fp8_to_bf16_dpas(
    %arg0: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<128x64xf8E4M3FN, {{.*}}> -> tensor<128x64xf16, {{.*}}>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<64x128xf8E5M2, {{.*}}> -> tensor<64x128xf16, {{.*}}>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<128x128xbf16, #[[$DPAS_MIXED_FP8_BF16]]> -> tensor<128x128xf32, #[[$DPAS_MIXED_FP8_BF16]]>
    // CHECK: %[[D:.+]] = tt.dot {{.*}} : tensor<128x64xf16{{.*}}> * tensor<64x128xf16{{.*}}> -> tensor<128x128xf32, #[[$DPAS_MIXED_FP8_BF16]]>
    // CHECK: tt.fp_to_fp %[[D]], rounding = rtne : tensor<128x128xf32, #[[$DPAS_MIXED_FP8_BF16]]> -> tensor<128x128xbf16, #[[$DPAS_MIXED_FP8_BF16]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Test mixed FP8 -> FP32 accumulation without FP8 DPAS (both extended to FP16, like same-format FP8)
// CHECK: #[[$DPAS_MIXED_FP8_FP32:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_mixed_fp8_to_fp32_dpas
  tt.func public @kernel_mixed_fp8_to_fp32_dpas(
    %arg0: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<128x64xf8E4M3FN, {{.*}}> -> tensor<128x64xf16, {{.*}}>
    // CHECK-DAG: tt.fp_to_fp {{.*}} : tensor<64x128xf8E5M2, {{.*}}> -> tensor<64x128xf16, {{.*}}>
    // CHECK: tt.dot {{.*}} : tensor<128x64xf16{{.*}}> * tensor<64x128xf16{{.*}}> -> tensor<128x128xf32, #[[$DPAS_MIXED_FP8_FP32]]>
    // CHECK-NOT: rounding
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
