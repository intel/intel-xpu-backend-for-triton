// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.intr.bitreverse

// Regression test for issue #6138: warp-dimension ConvertLayout must not
// produce sub-byte bitreverse patterns (llvm.bitreverse.i4) that crash IGC.
// The conversion changes warpsPerCTA from [1,2] to [2,1] — a warp swap that
// Intel handles via non-swizzled SLM to avoid the problematic XOR swizzle.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_layout_warp_swap
  tt.func @convert_layout_warp_swap(%arg0: tensor<4x32xf16, #blocked>) -> tensor<4x32xf16, #blocked1> {
    // CHECK: llvm.store {{.*}} !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj
    // CHECK: llvm.load {{.*}} !llvm.ptr<3>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<4x32xf16, #blocked> -> tensor<4x32xf16, #blocked1>
    tt.return %0 : tensor<4x32xf16, #blocked1>
  }
}
