// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Regression tests for large-stride bugs in the 2D-block-IO prefetch
// lowering (see intel/intel-xpu-backend-for-triton#7334):
//   1. getStride() truncated StrideInfo's int64 to int32, so a row stride of
//      2^31 became negative and the cooperative prefetch bailed out via
//      `if (stride < 0) return failure()`, emitting 0 prefetch ops.
//   2. In the rank>2 batch-stride loop, `dimStride * elemSizeInBytes` was an
//      int32-times-unsigned product, so a batch stride whose byte value
//      exceeded INT32_MAX truncated to 0 and every batch prefetched the
//      same surface.
//   3. getPitch() computed `(unsigned)stride * elemSizeInBits / 8` in 32-bit
//      unsigned arithmetic, so a stride whose byte pitch wrapped mod 2^32
//      to a value >= MIN_PITCH silently returned an i32 constant with the
//      truncated pitch. The new INT32_MAX guard returns null instead, so
//      the DPAS regular-pointer path bails out and the cooperative fallback
//      (which computes pitch via i64 mul + trunc) takes over.
//
// Cases 4 and 5 are in-range controls that exercise the ordinary path to
// prove the widened arithmetic and INT32_MAX guard don't break it.

// Case 1: row stride 2^31 (f16) — truncation bug on the cooperative path.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_bigstride_f16
  tt.func public @prefetch_bigstride_f16(%arg0: !tt.ptr<f16>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %e1 = arith.extsi %1 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
    %2 = arith.constant dense<2147483648> : tensor<128x1xi64, #blocked>
    %3 = arith.muli %e1, %2 : tensor<128x1xi64, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %e5 = arith.extsi %5 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %6 = tt.broadcast %3 : tensor<128x1xi64, #blocked> -> tensor<128x64xi64, #blocked>
    %7 = tt.broadcast %e5 : tensor<1x64xi64, #blocked> -> tensor<128x64xi64, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x64xi64, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %9, %8 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi64, #blocked>
    // CHECK-COUNT-2: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Case 2: rank-3, batch stride 2^32 (f16) — exercises the extra-dim stride
// loop at LoadStoreOpToLLVM.cpp:1784-1788, where `dimStride *
// elemSizeInBytes` is now widened to int64 via `static_cast<int64_t>`.
// Without the cast, an int32 * unsigned product would lose the high bits and
// synthesize aliasing per-batch offsets, silently prefetching the wrong
// surfaces. The inner (H, W) tile keeps small strides so only the batch
// dim carries the >INT32_MAX value.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_bigstride_rank3_f16
  tt.func public @prefetch_bigstride_rank3_f16(%arg0: !tt.ptr<f16>) {
    // Batch axis (dim 0): range [0..1] scaled by 2^32 -> batch stride = 2^32.
    %b0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3d}>}>>
    %b1 = tt.expand_dims %b0 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3d}>}>> -> tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked3d}>>
    %b2 = tt.expand_dims %b1 {axis = 2 : i32} : tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked3d}>> -> tensor<2x1x1xi32, #blocked3d>
    %be = arith.extsi %b2 : tensor<2x1x1xi32, #blocked3d> to tensor<2x1x1xi64, #blocked3d>
    %bs = arith.constant dense<4294967296> : tensor<2x1x1xi64, #blocked3d>
    %bmul = arith.muli %be, %bs : tensor<2x1x1xi64, #blocked3d>
    // Row axis (dim 1): stride = 64 elements.
    %r0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked3d}>}>>
    %r1 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked3d}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked3d}>>
    %r2 = tt.expand_dims %r1 {axis = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked3d}>> -> tensor<1x128x1xi32, #blocked3d>
    %re = arith.extsi %r2 : tensor<1x128x1xi32, #blocked3d> to tensor<1x128x1xi64, #blocked3d>
    %rs = arith.constant dense<64> : tensor<1x128x1xi64, #blocked3d>
    %rmul = arith.muli %re, %rs : tensor<1x128x1xi64, #blocked3d>
    // Col axis (dim 2): stride = 1.
    %c0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked3d}>}>>
    %c1 = tt.expand_dims %c0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked3d}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked3d}>>
    %c2 = tt.expand_dims %c1 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked3d}>> -> tensor<1x1x64xi32, #blocked3d>
    %ce = arith.extsi %c2 : tensor<1x1x64xi32, #blocked3d> to tensor<1x1x64xi64, #blocked3d>
    // Broadcast to the full 3D shape and combine.
    %bB = tt.broadcast %bmul : tensor<2x1x1xi64, #blocked3d> -> tensor<2x128x64xi64, #blocked3d>
    %rB = tt.broadcast %rmul : tensor<1x128x1xi64, #blocked3d> -> tensor<2x128x64xi64, #blocked3d>
    %cB = tt.broadcast %ce : tensor<1x1x64xi64, #blocked3d> -> tensor<2x128x64xi64, #blocked3d>
    %sum1 = arith.addi %bB, %rB : tensor<2x128x64xi64, #blocked3d>
    %sum2 = arith.addi %sum1, %cB : tensor<2x128x64xi64, #blocked3d>
    %splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x128x64x!tt.ptr<f16>, #blocked3d>
    %ptr = tt.addptr %splat, %sum2 : tensor<2x128x64x!tt.ptr<f16>, #blocked3d>, tensor<2x128x64xi64, #blocked3d>
    // The extra-dim (batch) stride in bytes must be `2^32 * sizeof(f16)` =
    // 8589934592. Without the int64 widening at LoadStoreOpToLLVM.cpp:1788,
    // this value would truncate to 0 and every batch would prefetch surface 0.
    // CHECK: llvm.mlir.constant(8589934592 : i64) : i64
    // CHECK-COUNT-4: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<2x128x64x!tt.ptr<f16>, #blocked3d>
    tt.return
  }
}

// -----

// Case 3: DPAS regular-pointer prefetch with row stride 2^30 + 16 f32
// elements (byte pitch = 4 * (2^30 + 16) = 2^32 + 64). Chosen so the
// buggy 32-bit unsigned multiply in getPitch() wraps to exactly 64
// (>= MIN_PITCH), which passes the "unsupported pitch" check and used
// to return `i32_val(64)`. The DPAS path would then emit prefetches
// with a bogus 64-byte pitch. With the INT32_MAX guard, getPitch()
// returns null, the DPAS path fails, and the cooperative fallback
// materializes pitch via i64 mul + trunc — so the pitch fed to the
// prefetch op is an `llvm.trunc ... : i64 to i32` rather than a bare
// i32 constant.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_getpitch_overflow_f32
  tt.func public @prefetch_getpitch_overflow_f32(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %e1 = arith.extsi %1 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> to tensor<64x1xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    // row stride = 2^30 + 16 elements; buggy `(unsigned)stride * 32 / 8` wraps to 64.
    %2 = arith.constant dense<1073741840> : tensor<64x1xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %3 = arith.muli %e1, %2 : tensor<64x1xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %e5 = arith.extsi %5 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> to tensor<1x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = tt.broadcast %3 : tensor<64x1xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %e5 : tensor<1x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = arith.addi %6, %7 : tensor<64x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %ptr = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi64, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    // Fixed getPitch bails on INT32_MAX overflow, so the DPAS regular-pointer
    // pattern fails and the cooperative fallback emits exactly one prefetch
    // for this warp's tile. The buggy code emitted 8 prefetches (one per
    // DPAS repetition) with a bogus base_pitch of 64 bytes.
    // CHECK-COUNT-1: triton_gen.2Dblockprefetch
    // CHECK-NOT: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}

// -----

// Case 4 (control): identical to case 1 but with a small (in-range) stride.
// Verifies that the widened-int64 code path does not break ordinary strides —
// a legitimately small stride/pitch still lowers to prefetch ops.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_control_f16
  tt.func public @prefetch_control_f16(%arg0: !tt.ptr<f16>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %e1 = arith.extsi %1 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
    %2 = arith.constant dense<64> : tensor<128x1xi64, #blocked>
    %3 = arith.muli %e1, %2 : tensor<128x1xi64, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %e5 = arith.extsi %5 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %6 = tt.broadcast %3 : tensor<128x1xi64, #blocked> -> tensor<128x64xi64, #blocked>
    %7 = tt.broadcast %e5 : tensor<1x64xi64, #blocked> -> tensor<128x64xi64, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x64xi64, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %9, %8 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi64, #blocked>
    // CHECK-COUNT-2: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Case 5 (control): DPAS regular-pointer prefetch with a small pitch
// (row stride = 32 f32 elements → pitch = 128 bytes). Confirms the
// getPitch() path still emits the pitch as an i32 constant when in range,
// exercising the INT32_MAX guard's non-overflow branch.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_pitch_ok_f32
  tt.func public @prefetch_pitch_ok_f32(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %2 = arith.constant dense<32> : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = arith.addi %6, %7 : tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %ptr = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    // CHECK: llvm.mlir.constant(128 : i32) : i32
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}
