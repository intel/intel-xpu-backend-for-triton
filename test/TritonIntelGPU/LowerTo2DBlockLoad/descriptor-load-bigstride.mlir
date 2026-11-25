// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

// Regression tests for large-stride bugs in the descriptor→2Dblockload
// lowering (companion to test/TritonIntelGPU/prefetch-bigstride.mlir; see
// intel/intel-xpu-backend-for-triton#7334).
//
// The 2Dblockload HW pitch/base_width operands are i32; the transform
// used to unconditionally truncate `stride * elemBytes` to i32, so a
// compile-time-foldable pitch stride exceeding INT32_MAX would silently
// produce a garbage surface descriptor. With the INT32_MAX guard, the
// transform now leaves the tt.descriptor_load unlowered (the HW verifier
// catches invalid pitches produced by runtime strides).

// Case 1: descriptor pitch stride = 2^30 f16 elements → byte pitch = 2^31,
// > INT32_MAX. The `arith.constant %stride` is a compile-time value the
// transform's `getFoldedConstantValue` can inspect. Expected: no
// ttig.2d_block_load emitted, and the tt.descriptor_load survives.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_overflow_f16
  tt.func @descriptor_load_pitch_overflow_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // Pitch stride: 2^30 f16 elements. Byte pitch = 2^30 * 2 = 2^31 (> INT32_MAX).
    %big_stride = arith.constant 1073741824 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%big_stride, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 2 (control): pitch stride = 64 f16 elements → byte pitch = 128,
// well within i32. Confirms the guard does not fire for legitimate strides.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_control_f16
  tt.func @descriptor_load_control_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%stride, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
