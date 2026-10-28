// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul="grf-mode=128" | FileCheck %s --check-prefix=GRF128

// COM: A 10 KB consumer copy fits the 256-entry GRF the default mode can escalate to, but not an explicit 128-entry GRF.
// CHECK: #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// CHECK-NOT: warpsPerCTA = [1, 4]
// GRF128-DAG: #[[$QK:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// GRF128-DAG: #[[$PV:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 4]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_grf_mode
  // GRF128-LABEL: chained_dot_decode_grf_mode
  tt.func public @chained_dot_decode_grf_mode(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked1>
    // GRF128: tt.dot {{.*}} -> tensor<4x32xf32, #[[$QK]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x32xf32, #blocked>
    %t = arith.truncf %d : tensor<4x32xf32, #blocked> to tensor<4x32xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x32xbf16, #blocked> -> tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // GRF128: tt.dot {{.*}} -> tensor<4x128xf32, #[[$PV]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x128xf32, #blocked1>
    tt.return %r : tensor<4x128xf32, #blocked1>
  }
}

// -----

// COM: More than 32 warps cannot escalate to a larger GRF mode, so the same 10 KB consumer copy goes along N.
// CHECK-DAG: #[[$QK:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [64, 1]{{.*}}>
// CHECK-DAG: #[[$PV:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 64]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 64], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 64], order = [1, 0]}>
module attributes {"ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_many_warps
  tt.func public @chained_dot_decode_many_warps(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<4x32xf32, #[[$QK]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x32xf32, #blocked>
    %t = arith.truncf %d : tensor<4x32xf32, #blocked> to tensor<4x32xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x32xbf16, #blocked> -> tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<4x128xf32, #[[$PV]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x128xf32, #blocked1>
    tt.return %r : tensor<4x128xf32, #blocked1>
  }
}

// -----

// COM: A target that escalates to 512 GRF entries holds a 24 KB consumer copy, so both dots stay on M.
// CHECK: #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// CHECK-NOT: warpsPerCTA = [1, 4]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.max_grf_mode" = "512", "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_max_grf_512
  tt.func public @chained_dot_decode_max_grf_512(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<16x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x512xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x16xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x512xf32, #blocked1>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x16xf32, #blocked>
    %t = arith.truncf %d : tensor<4x16xf32, #blocked> to tensor<4x16xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x16xbf16, #blocked> -> tensor<4x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x512xf32, #blocked1>
    tt.return %r : tensor<4x512xf32, #blocked1>
  }
}
