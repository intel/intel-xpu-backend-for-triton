// Tests for NaN-padded descriptor loads with non-4-byte-aligned inner shapes.
//
// Two cases:
//  (a) pitch < rounded_base_width  →  fallback to scalar (tt.descriptor_load
//      survives, NO ttig.2d_block_load emitted)
//  (b) pitch >= rounded_base_width →  base_width rounded up, ttig.2d_block_load
//      IS emitted with the rounded constant
//
// Background: PVC Max 1100 applies 2D block load OOB checks at i32 (4-byte)
// granularity.  base_width must be a multiple of 4 bytes (verifier constraint).
// For fp16 with 15 columns, base_width=30 bytes is not 4-aligned.  Rounding up
// to 32 is safe when pitch >= 32; otherwise the instruction would violate
// pitch >= base_width, so we fall back to scalar loads.

// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load \
// RUN:   | FileCheck %s

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // COM: (a) Contiguous fp16 tensor with 15 columns: pitch = 15*2 = 30 bytes.
  // COM: rounded_base_width = ceil(30/4)*4 = 32 bytes.  30 < 32 -> scalar fallback.
  // COM: The tt.descriptor_load must survive (not be converted to 2d_block_load).
  // CHECK-LABEL: tt.func @pad_nan_scalar_fallback
  // CHECK: tt.descriptor_load
  // CHECK-NOT: ttig.2d_block_load
  tt.func @pad_nan_scalar_fallback(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0> {
    %c15     = arith.constant 15  : i32
    %c15_i64 = arith.constant 15  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    // shape=[%m, 15], stride=[15, 1]  →  pitch = 15*2 = 30 bytes < rounded=32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c15], [%c15_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<64x16xf16, #dot0> -> tensor<64x16xf16, #dot0>
    tt.return %0 : tensor<64x16xf16, #dot0>
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // COM: (b) Padded fp16 tensor: 15 valid columns but pitch stride = 16 elements.
  // COM: pitch = 16*2 = 32 bytes >= rounded_base_width = 32 bytes.
  // COM: The pass must emit ttig.2d_block_load with base_width=32 (not 30).
  // CHECK-LABEL: tt.func @pad_nan_rounded_base_width
  // CHECK-NOT: tt.descriptor_load
  // CHECK: arith.constant 32 : i32
  // CHECK: ttig.2d_block_load
  // CHECK-SAME: {row_major, pad_nan}
  tt.func @pad_nan_rounded_base_width(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0> {
    %c15     = arith.constant 15  : i32
    %c16_i64 = arith.constant 16  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    // shape=[%m, 15], stride=[16, 1]  →  pitch = 16*2 = 32 bytes >= rounded=32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c15], [%c16_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<64x16xf16, #dot0> -> tensor<64x16xf16, #dot0>
    tt.return %0 : tensor<64x16xf16, #dot0>
  }
}
