// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// Test FP16 -> FP16 accumulation
// CHECK: #[[$DPAS:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_fp16_to_fp16
  tt.func public @kernel_fp16_to_fp16(
    %arg0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: tt.dot {{.*}} : tensor<128x64xf16{{.*}}> * tensor<64x128xf16{{.*}}> -> tensor<128x128xf16, #[[$DPAS]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf16, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Test BF16 -> BF16 accumulation
// CHECK: #[[$DPAS2:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_bf16_to_bf16
  tt.func public @kernel_bf16_to_bf16(
    %arg0: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK: tt.dot {{.*}} : tensor<128x64xbf16{{.*}}> * tensor<64x128xbf16{{.*}}> -> tensor<128x128xbf16, #[[$DPAS2]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Test FP8 -> BF16 accumulation (Xe3P+)
// CHECK-NOT: ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate", "ttig.support_subgroup_matrix_multiply_accumulate_bf8"} {
  // CHECK-LABEL: tt.func public @kernel_fp8_to_bf16_xe3p
  tt.func public @kernel_fp8_to_bf16_xe3p(
    %arg0: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // FP8 types get converted to BF16, so no DPAS is used
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Test mixed types (FP16 x BF16 -> FP32, should NOT use DPAS)
// CHECK-NOT: ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_fp16_bf16_mixed
  tt.func public @kernel_fp16_bf16_mixed(
    %arg0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Mixed FP16 and BF16 operands should not use DPAS
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
