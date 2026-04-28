// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=ttig.2d_block_ptr_load --implicit-check-not=ttig.2d_block_load

// COM: Pointer-based load from a function arg has unknown stride, so it
// COM: cannot be converted (stride analysis returns -1). Stays as tt.load.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @ptr_load_unknown_stride
  tt.func @ptr_load_unknown_stride(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0> {
    // CHECK: tt.load
    %0 = tt.load %arg0 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Load without block_io attribute should NOT be converted.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @no_block_io_attr
  tt.func @no_block_io_attr(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0> {
    // CHECK: tt.load
    %0 = tt.load %arg0 : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Module without support_2d_block_io attribute should skip the pass entirely.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_2d_block_support
  tt.func @no_2d_block_support(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0> {
    // CHECK: tt.load
    %0 = tt.load %arg0 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
