// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Regression tests for large-stride bugs in the descriptor→2Dblockstore
// lowering (companion to test/TritonIntelGPU/prefetch-bigstride.mlir; see
// intel/intel-xpu-backend-for-triton#7334).
//
// The descriptor-store lowering path computes baseWidth and pitch as
// `mul(i64_stride, i64_val(elemBytes)) : i64` followed by `trunc i64 to i32`.
// Without the narrowSurfaceBytesOrNull() guard, a compile-time-foldable
// pitch/shape whose byte value exceeds the HW's 24-bit field silently
// produces a garbage surface descriptor. With the guard, the pattern
// returns failure() and the tt.descriptor_store is left in place (or
// lowered via another pattern).

// Case 1: pitch stride = 2^30 f16 elements → byte pitch = 2^31, well over
// the HW's 24-bit limit. Expected: no triton_gen.2Dblockstore emitted
// (pattern bails cleanly).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_pitch_overflow_f16
  tt.func public @store_pitch_overflow_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // pitch stride = 2^30 f16 → 2^31 bytes (overflows i32).
    %big_stride = arith.constant 1073741824 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%big_stride, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.return
  }
}

// -----

// Case 2 (control): pitch stride = 64 f16 elements → byte pitch = 128,
// well within i32. Confirms the guard does not fire for legitimate strides
// and the ordinary store lowering emits triton_gen.2Dblockstore ops.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_control_f16
  tt.func public @store_control_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %stride = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%stride, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.return
  }
}
