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

// Test per-operation shape selection with DPAS-compatible operand types.
// CHECK: #[[$SHAPE_DPAS:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_native_k_dpas
  tt.func public @kernel_native_k_dpas(
    %a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %out: tensor<16x16x!tt.ptr<f32>, #blocked>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[$SHAPE_DPAS]]>
    %result = tt.dot %a, %b, %acc : tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.store %out, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @kernel_small_n_fma
  tt.func public @kernel_small_n_fma(
    %a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %out: tensor<16x8x!tt.ptr<f32>, #blocked>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x8xf32, #blocked>
    %result = tt.dot %a, %b, %acc : tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x8xf32, #blocked>
    tt.store %out, %result : tensor<16x8x!tt.ptr<f32>, #blocked>
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
