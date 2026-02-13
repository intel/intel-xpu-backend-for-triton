// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// Test INT8 signed dot operations (S8_S8 -> S32)
// CHECK: #[[$DPAS:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_int8_signed
  tt.func public @kernel_int8_signed(
    %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<i32>, #blocked>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    // CHECK: tt.dot {{.*}} : tensor<128x64xi8{{.*}}> * tensor<64x128xi8{{.*}}> -> tensor<128x128xi32, #[[$DPAS]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xi32, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// Test INT8 with different configurations to ensure DPAS is used correctly
// CHECK: #[[$DPAS2:.+]] = #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_int8_alt_config
  tt.func public @kernel_int8_alt_config(
    %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<i32>, #blocked>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    // CHECK: tt.dot {{.*}} : tensor<128x64xi8{{.*}}> * tensor<64x128xi8{{.*}}> -> tensor<128x128xi32, #[[$DPAS2]]>
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xi32, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}// -----

// Test INT8 with wrong output width (should NOT use DPAS - needs i32, got i16)
// CHECK-NOT: ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: tt.func public @kernel_int8_wrong_output
  tt.func public @kernel_int8_wrong_output(
    %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<i16>, #blocked>) {
    %cst = arith.constant dense<0> : tensor<128x128xi16, #blocked>
    // INT8 -> INT16 not supported by DPAS (needs INT32 accumulator)
    %result = tt.dot %arg0, %arg1, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xi16, #blocked>
    %1 = ttg.convert_layout %result : tensor<128x128xi16, #blocked> -> tensor<128x128xi16, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<i16>, #blocked>
    tt.return
  }
}
