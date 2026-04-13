// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.intr.bitreverse

// Regression test for issue #6138: warp-dimension ConvertLayout must not
// produce sub-byte bitreverse patterns (llvm.bitreverse.i4) that crash IGC.
// Intel handles these conversions via non-swizzled SLM (flat addressing)
// instead of falling through to the upstream NVIDIA-oriented XOR swizzle.

// Test 1: warpsPerCTA swap [1,2] -> [2,1] (the original crash case)
#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_layout_warp_swap_small
  tt.func @convert_layout_warp_swap_small(%arg0: tensor<4x32xf16, #blocked>) -> tensor<4x32xf16, #blocked1> {
    // CHECK: llvm.store {{.*}} !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj
    // CHECK: llvm.load {{.*}} !llvm.ptr<3>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<4x32xf16, #blocked> -> tensor<4x32xf16, #blocked1>
    tt.return %0 : tensor<4x32xf16, #blocked1>
  }
}

// -----

// Test 2: Larger tile — warp swap with more warps
#blocked2  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_layout_warp_swap_large
  tt.func @convert_layout_warp_swap_large(%arg0: tensor<8x64xf32, #blocked2>) -> tensor<8x64xf32, #blocked3> {
    // CHECK: llvm.store {{.*}} !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj
    // CHECK: llvm.load {{.*}} !llvm.ptr<3>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<8x64xf32, #blocked2> -> tensor<8x64xf32, #blocked3>
    tt.return %0 : tensor<8x64xf32, #blocked3>
  }
}
