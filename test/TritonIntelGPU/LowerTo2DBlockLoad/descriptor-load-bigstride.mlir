// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

// Regression tests for the surface-parameter bounds check in the
// descriptor→2Dblockload lowering (companion to
// test/TritonIntelGPU/prefetch-bigstride.mlir; see
// intel/intel-xpu-backend-for-triton#7334 and #8071).
//
// The 2Dblockload HW encodes base_width (bytes), base_height (rows) and
// base_pitch (bytes) as 24-bit "value - 1" fields, so each must lie in
// [1, 2^24]. For every candidate tt.make_tensor_descriptor, the transform
// leaves the tt.descriptor_load unlowered if a field that is a compile-time
// constant falls outside that range:
//   width  = shape[descRank-1] * elemBytes   (the column dim)
//   height = shape[descRank-2]               (the row dim)
//   pitch  = stride[descRank-2] * elemBytes  (the row dim)
// An operand is a compile-time constant only if getFoldedConstantValue
// resolves it to a literal constant op; it does not evaluate arithmetic, so
// e.g. an arith.addi of two arith.constants is a runtime value. Runtime values
// are trusted. The byte products are overflow-safe, so a stride whose product
// wraps int64 back into range is still rejected. The batch dims of a
// rank-reducing descriptor are not surface fields and are not range-checked
// (Case 20). Each negative case also checks that the pass emitted no IR before
// bailing (no stray ttig.extract_desc).
//
// This pass deliberately enforces neither the 64 B minimum nor the 16 B
// alignment; both are left to a follow-up issue, and no case in this file
// pins either behavior. @descriptor_load_rank_reducing in descriptor-load.mlir
// (2 B pitch) is an existing test that must keep converting.
//
// From Case 3 on, at most one width/height/pitch operand is a compile-time
// constant (the field under test; none in Case 20) and the others are runtime
// arguments. So are the remaining operands, except the innermost stride
// (always %c1_i64) and the batch extent of the rank-reducing cases (plus the
// batch stride in Case 20). A check on the wrong operand therefore fails the
// test.

// Case 1: descriptor pitch stride = 2^30 f16 elements → byte pitch = 2^31,
// which exceeds 2^24. `%big_stride` is an `arith.constant`, which the
// transform's `getFoldedConstantValue` can inspect. Expected: no
// ttig.2d_block_load emitted, and the tt.descriptor_load survives.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_overflow_f16
  tt.func @descriptor_load_pitch_overflow_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // Pitch stride: 2^30 f16 elements. Byte pitch = 2^30 * 2 = 2^31 (> 2^24).
    %big_stride = arith.constant 1073741824 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%big_stride, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 2 (control): pitch stride = 64 f16 elements → byte pitch = 128,
// well within [1, 2^24]. Confirms the guard does not fire for legitimate strides.
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

// -----

// Case 3: height = 2^24 rows, the largest value the 24-bit field encodes; the upper bound is inclusive.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_height_max_f16
  tt.func @descriptor_load_height_max_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %height = arith.constant 16777216 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%height, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 4: height = 1 row, the smallest legal value; the lower bound is inclusive.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_height_one_f16
  tt.func @descriptor_load_height_one_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %height = arith.constant 1 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%height, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 5: width = 2^23 f16 elements = 2^24 B, the largest legal byte width; the upper bound is inclusive.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_width_max_f16
  tt.func @descriptor_load_width_max_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %width = arith.constant 8388608 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %width], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 6 (control): f32 pitch stride 64 = 256 B is in range; baseline for the f32 pitch cases (and the value Case 16 wraps to).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_control_f32
  tt.func @descriptor_load_pitch_control_f32(%arg0: !tt.ptr<f32>, %argH: i32, %argW: i32) -> tensor<64x32xf32, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f32>, <64x32xf32>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32, #dot0>
    tt.return %0 : tensor<64x32xf32, #dot0>
  }
}

// -----

// Case 7: f32 pitch stride 2^22 = 2^24 B, the largest legal pitch; with Case 15 this pins the 4-byte element scale.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_max_f32
  tt.func @descriptor_load_pitch_max_f32(%arg0: !tt.ptr<f32>, %argH: i32, %argW: i32) -> tensor<64x32xf32, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 4194304 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f32>, <64x32xf32>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32, #dot0>
    tt.return %0 : tensor<64x32xf32, #dot0>
  }
}

// -----

// Case 8: height = 2^24 + 1 rows, one past what the 24-bit value - 1 field can encode.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_height_over_f16
  tt.func @descriptor_load_height_over_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %height = arith.constant 16777217 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%height, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 9: height = 0 rows is below the lower bound; the value - 1 field would wrap it to 2^24 - 1.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_height_zero_f16
  tt.func @descriptor_load_height_zero_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %height = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%height, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 10: height = -1 is out of range; a negative shape must be rejected, not truncated into the 24-bit field.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_height_neg_f16
  tt.func @descriptor_load_height_neg_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %height = arith.constant -1 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%height, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 11: width = 0 B is below the lower bound; the value - 1 field would wrap it to 2^24 - 1.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_width_zero_f16
  tt.func @descriptor_load_width_zero_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %width = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %width], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 12: width = (2^23 + 1) f16 elements = 2^24 + 2 B, just past the byte bound (Case 5 is the last legal value).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_width_over_f16
  tt.func @descriptor_load_width_over_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argP: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %width = arith.constant 8388609 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %width], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 13: pitch = 0 B is below the lower bound; the value - 1 field would wrap it to 2^24 - 1.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_zero_f16
  tt.func @descriptor_load_pitch_zero_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 0 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 14: pitch stride -64 f16 = -128 B is out of range; a negative pitch must be rejected, not truncated.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_neg_f16
  tt.func @descriptor_load_pitch_neg_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant -64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 15: f32 pitch stride 2^22 + 1 = 2^24 + 4 B, just past the byte bound; with Case 7 this pins the 4-byte element scale.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_over_f32
  tt.func @descriptor_load_pitch_over_f32(%arg0: !tt.ptr<f32>, %argH: i32, %argW: i32) -> tensor<64x32xf32, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 4194305 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f32>, <64x32xf32>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32, #dot0>
    tt.return %0 : tensor<64x32xf32, #dot0>
  }
}

// -----

// Case 16: f32 pitch stride 2^62 + 64; stride * 4 wraps int64 to exactly 256 B (in range, cf. Case 6), so only an overflow-safe product rejects it.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pitch_i64_wrap_f32
  tt.func @descriptor_load_pitch_i64_wrap_f32(%arg0: !tt.ptr<f32>, %argH: i32, %argW: i32) -> tensor<64x32xf32, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 4611686018427387968 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%stride, %c1_i64] : <f32>, <64x32xf32>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32, #dot0>
    tt.return %0 : tensor<64x32xf32, #dot0>
  }
}

// -----

// Case 17: rank-reducing load (1x64x32 desc -> 64x32 result); the height is desc dim 1 (= 0), not dim 0 (the valid batch of 1).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_rank_reducing_height_zero_f16
  tt.func @descriptor_load_rank_reducing_height_zero_f16(%arg0: !tt.ptr<f16>, %argW: i32, %argB: i64, %argP: i64, %b: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %c0_i32, %argW], [%argB, %argP, %c1_i64] : <f16>, <1x64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%b, %c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 18: scf.if yields one of two descriptors and only the then-branch one has height 0; every candidate must be checked.
// Candidates are collected LIFO (findAllMakeTensorDescOps pops the else-yield first), so this case catches a check that
// inspects only the first candidate.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_if_bad_then_f16
  tt.func @descriptor_load_if_bad_then_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %cond: i1) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%c0_i32, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 19: mirror of Case 18 with the height-0 descriptor in the else-branch, so the test does not depend on candidate order.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_if_bad_else_f16
  tt.func @descriptor_load_if_bad_else_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %cond: i1) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%c0_i32, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 20: rank-reducing load (1x64x32 desc -> 64x32 result) with batch extent 2^24 + 1 and batch stride 2^40 f16 elements;
// width, height and pitch are runtime values, so the load must convert. Batch dims are not surface fields and are not
// range-checked: the batch stride is folded into the base pointer and the batch extent is passed as batch_shapes.
// This catches field indices that use the result rank instead of descRank throughout: the height check would then read
// the batch extent and bail. Case 17 cannot catch that uniform variant, because its mutated width index reads the bad
// height (0) and bails for the wrong reason. It also catches a pitch index that drops the (descRank - 2) offset, which
// would read the 2^40 batch stride.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_rank_reducing_big_batch_f16
  tt.func @descriptor_load_rank_reducing_big_batch_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %b: i32) -> tensor<64x32xf16, #dot0> {
    %c16777217_i32 = arith.constant 16777217 : i32
    %c1099511627776_i64 = arith.constant 1099511627776 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c16777217_i32, %argH, %argW], [%c1099511627776_i64, %argP, %c1_i64] : <f16>, <1x64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%b, %c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 21: as Case 18, but for pitch: only the then-branch descriptor has pitch stride 0, so the pitch check must also
// inspect every candidate.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_if_bad_pitch_then_f16
  tt.func @descriptor_load_if_bad_pitch_then_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %cond: i1) -> tensor<64x32xf16, #dot0> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 22: mirror of Case 21 with the pitch-0 descriptor in the else-branch.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_if_bad_pitch_else_f16
  tt.func @descriptor_load_if_bad_pitch_else_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %cond: i1) -> tensor<64x32xf16, #dot0> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%argH, %argW], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 23: scf.for with the load on the loop-carried descriptor inside the body. The iter_arg traces to both the
// (valid) init and the yielded descriptor, which has height 0, so the load must bail.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_for_iter_arg_bad_yield_height_f16
  tt.func @descriptor_load_for_iter_arg_bad_yield_height_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %N: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc_init = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %result = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%desc = %desc_init) -> (!tt.tensordesc<64x32xf16>) : i32 {
      %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
      %new_desc = tt.make_tensor_descriptor %arg0, [%c0_i32, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %new_desc : !tt.tensordesc<64x32xf16>
    }
    tt.return
  }
}

// -----

// Case 24: scf.for with the load on the loop result after the loop. The result traces to the yielded descriptor,
// which has width 0, so the load must bail.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_for_result_bad_yield_width_f16
  tt.func @descriptor_load_for_result_bad_yield_width_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %N: i32) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc_init = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    %result = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%desc = %desc_init) -> (!tt.tensordesc<64x32xf16>) : i32 {
      %new_desc = tt.make_tensor_descriptor %arg0, [%argH, %c0_i32], [%argP, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %new_desc : !tt.tensordesc<64x32xf16>
    }
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %result[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 25: scf.while with the load on the after-region argument. It traces through scf.condition to the
// before-region argument, and from there to the (valid) init and the after-region yield, which has pitch stride 0.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_while_bad_yield_pitch_f16
  tt.func @descriptor_load_while_bad_yield_pitch_f16(%arg0: !tt.ptr<f16>, %argH: i32, %argW: i32, %argP: i64, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc_init = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%argP, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load
    %res = scf.while (%d = %desc_init) : (!tt.tensordesc<64x32xf16>) -> !tt.tensordesc<64x32xf16> {
      scf.condition(%cond) %d : !tt.tensordesc<64x32xf16>
    } do {
    ^bb0(%d2: !tt.tensordesc<64x32xf16>):
      %load = tt.descriptor_load %d2[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
      %new_desc = tt.make_tensor_descriptor %arg0, [%argH, %argW], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %new_desc : !tt.tensordesc<64x32xf16>
    }
    tt.return
  }
}
