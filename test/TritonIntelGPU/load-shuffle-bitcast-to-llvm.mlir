// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#src = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.linear<{register = [[128], [2], [4]], lane = [[1], [8], [16], [32], [64]], warp = [[256], [512]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: llvm.func @load_shuffle_bitcast_tensor_ptrs
  // CHECK-NOT: ttig.load_shuffle_bitcast
  // CHECK-NOT: tt.load
  // CHECK-NOT: ttg.convert_layout
  // CHECK: llvm.load
  tt.func public @load_shuffle_bitcast_tensor_ptrs(
      %arg0: tensor<1024x!tt.ptr<f16>, #src>) {
    %0 = ttig.load_shuffle_bitcast %arg0 : tensor<1024x!tt.ptr<f16>, #src> -> tensor<1024xf16, #dst>
    tt.return
  }
}
