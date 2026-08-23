// Tests for boundary-padded descriptor loads with non-4-byte-aligned inner
// shapes. Both PAD_NAN and PAD_ZERO descriptors take this path.
//
// Pitch-vs-rounding cases:
//  (a) pitch < rounded_base_width  →  fallback to scalar (tt.descriptor_load
//      survives, NO ttig.2d_block_load emitted)
//  (b) pitch >= rounded_base_width →  ttig.2d_block_load IS emitted.
//      base_width stays at the ORIGINAL (unrounded) value in the op.
//      LoadStoreOpToLLVM.cpp applies the rounding for the hardware instruction.
//
// Background: PVC Max 1100 applies 2D block load OOB checks at i32 (4-byte)
// granularity.  base_width must be a multiple of 4 bytes (verifier constraint).
// For fp16 with 15 columns, base_width=30 bytes is not 4-aligned.  Rounding up
// to 32 is safe when pitch >= 32; otherwise the instruction would violate
// pitch >= base_width, so we fall back to scalar loads.
//
// Additional cases:
//  (c) int8 B operand VNNI with K_rows=2: must fall back to scalar because
//      int8 packs 4 rows per i32 word (granularity 4/1=4) and 2 % 4 != 0.
//      Fixed: outerShapeAligned uses kAlignBytes/elemSizeBytes as granularity.
//  (d) Row-major A operand with compile-time odd row count: must emit
//      ttig.2d_block_load (A has no VNNI row-pairing issue).
//      Fixed: outerShapeAligned check restricted to VNNI B operand (opIdx==1).
//  (e) PAD_ZERO with a misaligned column count: `pad_zero` IS set, because the
//      lowering rounds base_width up and the mask must restore the zero fill.
//  (f) PAD_ZERO with an aligned column count: `pad_zero` is NOT set. No
//      rounding happens, so the hardware's own zero fill is already exact and
//      the attribute would only add dead arithmetic. PAD_ZERO is the default
//      for `tt.make_tensor_descriptor`, so this is the common path.
//  (g) PAD_NAN with an aligned column count: `pad_nan` IS still set. Hardware
//      never fills NaN, so this mask is unconditional — the asymmetry with (f)
//      is deliberate.
//  (h)/(i) Non-zero column index: 64-byte alignment compensation widens
//      base_width by `ptr & 0x3F` before rounding, so the pitch must clear
//      roundedBytes + kAlignCompBound (65) rather than just roundedBytes.

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
  // COM: The pass emits ttig.2d_block_load with base_width = ORIGINAL 30 bytes
  // COM: (15 elements * 2 bytes). Rounding to 32 happens in LoadStoreOpToLLVM.
  // COM: This also validates that the pitch-aware path is taken (not scalar).
  // CHECK-LABEL: tt.func @pad_nan_rounded_base_width
  // CHECK-NOT: tt.descriptor_load
  // Capture the base_width SSA value (result of the muli on the extracted shape).
  // This verifies the transform pass emits the ORIGINAL unrounded width, not a
  // premature arith.constant 32 : i32.
  // CHECK: %[[BW:.*]] = arith.muli
  // CHECK: ttig.2d_block_load %{{.*}}, %[[BW]],
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

// -----

// Case (c): int8 B operand (VNNI) with K_rows=2.
// The outerShapeAligned check uses vnniGranularity = kAlignBytes/elemSizeBytes.
// For int8 (1 byte): 4/1=4. K_rows=2 % 4 != 0, so the hardware may zero rows
// 0-1 as part of the OOB coarse i32 check.
// Correct behavior: fall back to scalar (tt.descriptor_load survives).
// Fix: outerShapeAligned now uses element-size-aware granularity (Issue 4).

// CHECK-LABEL: tt.func @pad_nan_int8_vnni_k2_should_fallback
// CHECK: tt.descriptor_load
// CHECK-NOT: ttig.2d_block_load

#dpas_i8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dot1_i8 = #ttg.dot_op<{opIdx = 1, parent = #dpas_i8, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_nan_int8_vnni_k2_should_fallback(%arg0: !tt.ptr<i8>, %n: i32) -> tensor<32x16xi8, #dot1_i8> {
    %c2     = arith.constant 2   : i32  // K_rows = 2: even by %2, odd by %4
    %c16_i64 = arith.constant 16 : i64  // pitch stride = 16 elements
    %c1_i64  = arith.constant 1  : i64
    %c0_i32  = arith.constant 0  : i32
    // B descriptor: shape=[2, %n], stride=[16, 1] -> pitch=16 bytes >= rounded(2)=4
    // K_rows=2: 2 % 2 == 0 passes current check (BUG), 2 % 4 != 0 should fail.
    %desc = tt.make_tensor_descriptor %arg0, [%c2, %n], [%c16_i64, %c1_i64]
              {padding = 2 : i32} : <i8>, !tt.tensordesc<32x16xi8, #dot1_i8>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<32x16xi8, #dot1_i8> -> tensor<32x16xi8, #dot1_i8>
    tt.return %0 : tensor<32x16xi8, #dot1_i8>
  }
}

// -----

// Case (d): row-major A operand with compile-time odd row count (M-1 = 127).
// The outer shape check fires for ALL padNan/padZero loads including row-major A,
// but the i32-pairing problem only exists for VNNI (B operand). Row-major A has
// no row packing — an odd row count is fine and must NOT cause scalar fallback.
// Correct behavior: ttig.2d_block_load IS emitted.
// BUG: current code falls back to scalar (% 2 fires on M-1=127).
// Fix: outerShapeAligned check now restricted to VNNI B operand (opIdx==1) (Issue 5).

// CHECK-LABEL: tt.func @pad_nan_row_major_a_odd_row_count_should_not_fallback
// CHECK-NOT: tt.descriptor_load
// CHECK: ttig.2d_block_load
// CHECK-SAME: {row_major, pad_nan}

#dpas_ra = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_ra = #ttg.dot_op<{opIdx = 0, parent = #dpas_ra, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_nan_row_major_a_odd_row_count_should_not_fallback(%arg0: !tt.ptr<f16>) -> tensor<128x16xf16, #dot0_ra> {
    %c127    = arith.constant 127 : i32  // M-1 = 127 (odd): row-major A, no VNNI
    %c15     = arith.constant 15  : i32  // N-1 = 15 (misaligned col count)
    %c16_i64 = arith.constant 16  : i64  // pitch stride = 16 elements
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    // A descriptor: shape=[127, 15], stride=[16, 1] -> pitch=32 >= rounded(30)=32
    // Row count 127 is odd: outer shape check INCORRECTLY causes scalar fallback
    // for A because A has no VNNI row-pairing — the check should not apply here.
    %desc = tt.make_tensor_descriptor %arg0, [%c127, %c15], [%c16_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<128x16xf16, #dot0_ra>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<128x16xf16, #dot0_ra> -> tensor<128x16xf16, #dot0_ra>
    tt.return %0 : tensor<128x16xf16, #dot0_ra>
  }
}

// -----

// Case (e): PAD_ZERO (padding = 1) with a misaligned column count.
// 15 fp16 columns = 30 bytes, not 4-aligned, pitch = 16*2 = 32 >= rounded 32.
// LoadStoreOpToLLVM rounds base_width 30 -> 32, which makes the hardware treat
// the two padding bytes as in-bounds. `pad_zero` must be set so the software
// mask restores the zero fill the hardware would otherwise have applied.

// CHECK-LABEL: tt.func @pad_zero_misaligned_sets_pad_zero
// CHECK-NOT: tt.descriptor_load
// CHECK: ttig.2d_block_load
// CHECK-SAME: {row_major, pad_zero}

#dpas_ez = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_ez = #ttg.dot_op<{opIdx = 0, parent = #dpas_ez, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_zero_misaligned_sets_pad_zero(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0_ez> {
    %c15     = arith.constant 15  : i32
    %c16_i64 = arith.constant 16  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c15], [%c16_i64, %c1_i64]
              {padding = 1 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0_ez>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
         : !tt.tensordesc<64x16xf16, #dot0_ez> -> tensor<64x16xf16, #dot0_ez>
    tt.return %0 : tensor<64x16xf16, #dot0_ez>
  }
}

// -----

// Case (f): PAD_ZERO with an ALIGNED column count.
// 16 fp16 columns = 32 bytes, already 4-aligned, so base_width is never
// rounded and the hardware zero-fills out-of-bounds elements exactly.
// `pad_zero` must NOT be set — otherwise every default-padding descriptor load
// (PAD_ZERO is the `tt.make_tensor_descriptor` default) pays for base_width
// rounding arithmetic plus a per-register zero-mask select for nothing.

// CHECK-LABEL: tt.func @pad_zero_aligned_omits_pad_zero
// CHECK-NOT: tt.descriptor_load
// CHECK: ttig.2d_block_load
// CHECK-SAME: {row_major}
// CHECK-NOT: pad_zero

#dpas_fz = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_fz = #ttg.dot_op<{opIdx = 0, parent = #dpas_fz, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_zero_aligned_omits_pad_zero(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0_fz> {
    %c16     = arith.constant 16  : i32
    %c16_i64 = arith.constant 16  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c16], [%c16_i64, %c1_i64]
              {padding = 1 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0_fz>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
         : !tt.tensordesc<64x16xf16, #dot0_fz> -> tensor<64x16xf16, #dot0_fz>
    tt.return %0 : tensor<64x16xf16, #dot0_fz>
  }
}

// -----

// Case (g): PAD_NAN with an ALIGNED column count.
// Counterpart to (f): `pad_nan` is unconditional because the hardware fills
// out-of-bounds elements with zero, never NaN, so the mask is always required
// regardless of base_width alignment.

// CHECK-LABEL: tt.func @pad_nan_aligned_keeps_pad_nan
// CHECK-NOT: tt.descriptor_load
// CHECK: ttig.2d_block_load
// CHECK-SAME: {row_major, pad_nan}

#dpas_gn = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_gn = #ttg.dot_op<{opIdx = 0, parent = #dpas_gn, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_nan_aligned_keeps_pad_nan(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0_gn> {
    %c16     = arith.constant 16  : i32
    %c16_i64 = arith.constant 16  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c16], [%c16_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0_gn>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<64x16xf16, #dot0_gn> -> tensor<64x16xf16, #dot0_gn>
    tt.return %0 : tensor<64x16xf16, #dot0_gn>
  }
}

// -----

// Case (h): non-zero column index, pitch just BELOW the required headroom.
// A non-zero column index means LoadStoreOpToLLVM's 64-byte alignment
// compensation can widen base_width by up to 63 bytes (`ptr & 0x3F`) BEFORE
// rounding, so the pitch has to clear roundedBytes + kAlignCompBound, not just
// roundedBytes.
//   colBytes = 63*2 = 126, roundedBytes = 128, minPitch = 128 + 65 = 193
//   pitch = 96*2 = 192  <  193  ->  scalar fallback
// This pins kAlignCompBound: with the plain `pitch >= roundedBytes` check this
// load would be emitted and could violate base_width <= pitch at runtime.

// CHECK-LABEL: tt.func @pad_nan_nonzero_col_idx_tight_pitch_falls_back
// CHECK: tt.descriptor_load
// CHECK-NOT: ttig.2d_block_load

#dpas_hn = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_hn = #ttg.dot_op<{opIdx = 0, parent = #dpas_hn, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_nan_nonzero_col_idx_tight_pitch_falls_back(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0_hn> {
    %c63     = arith.constant 63  : i32
    %c96_i64 = arith.constant 96  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    %c16_i32 = arith.constant 16  : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c63], [%c96_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0_hn>
    %0 = tt.descriptor_load %desc[%c0_i32, %c16_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<64x16xf16, #dot0_hn> -> tensor<64x16xf16, #dot0_hn>
    tt.return %0 : tensor<64x16xf16, #dot0_hn>
  }
}

// -----

// Case (i): same as (h) but pitch just ABOVE the required headroom.
//   colBytes = 63*2 = 126, roundedBytes = 128, minPitch = 128 + 65 = 193
//   pitch = 97*2 = 194  >=  193  ->  ttig.2d_block_load IS emitted
// Together with (h) this brackets minPitch exactly: 192 rejects, 194 accepts,
// and 193 is unreachable for a 2-byte element type.

// CHECK-LABEL: tt.func @pad_nan_nonzero_col_idx_sufficient_pitch_emits
// CHECK-NOT: tt.descriptor_load
// CHECK: ttig.2d_block_load
// CHECK-SAME: {row_major, pad_nan}

#dpas_in = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_in = #ttg.dot_op<{opIdx = 0, parent = #dpas_in, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @pad_nan_nonzero_col_idx_sufficient_pitch_emits(%arg0: !tt.ptr<f16>, %m: i32) -> tensor<64x16xf16, #dot0_in> {
    %c63     = arith.constant 63  : i32
    %c97_i64 = arith.constant 97  : i64
    %c1_i64  = arith.constant 1   : i64
    %c0_i32  = arith.constant 0   : i32
    %c16_i32 = arith.constant 16  : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %c63], [%c97_i64, %c1_i64]
              {padding = 2 : i32} : <f16>, !tt.tensordesc<64x16xf16, #dot0_in>
    %0 = tt.descriptor_load %desc[%c0_i32, %c16_i32]
           {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}
         : !tt.tensordesc<64x16xf16, #dot0_in> -> tensor<64x16xf16, #dot0_in>
    tt.return %0 : tensor<64x16xf16, #dot0_in>
  }
}
