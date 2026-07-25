// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Regression test for https://github.com/intel/intel-xpu-backend-for-triton/pull/7584
//
// The 2D block-store lowering previously applied:
//   adjustedBaseWidth = umax(adjustedBaseWidth, 64)
// This inflated base_width to 64 even when the natural tile base_width was
// smaller (e.g. 32 bytes for f16 with tile_width=16). Combined with TritonGEN's
// 64-byte pointer alignment compensation, the effective HW base_width could
// exceed the surface pitch, violating the pitch >= base_width invariant and
// producing incorrect results.
//
// Test: f16 DPAS C=[8,16] with warpsPerCTA=[1,2].  The column offset comes
// from the warp ID (a runtime subgroup-id call), making adjustedBaseWidth a
// runtime expression that the TritonGEN verifier cannot check at compile time.
// The fix removes the umax, so no llvm.umax instruction must appear in the
// generated code.
//
//   warp 0: offsetX = 0  → adjustedBaseWidth_natural = 32 bytes
//   warp 1: offsetX = 16 → adjustedBaseWidth_natural = 64 bytes
// Row stride = 32 elements → pitch = 64 bytes (HW minimum; constant).

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_no_umax_on_base_width
  // The adjustedBaseWidth must NOT be padded with umax(., 64).
  // CHECK-NOT: llvm.umax
  // The 2D block store must be emitted — block IO conversion must succeed.
  // CHECK: triton_gen.2Dblockstore
  // No umax anywhere in the function (not just before the store).
  // CHECK-NOT: llvm.umax
  tt.func public @store_no_umax_on_base_width(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x32xf16, #dpas>
    // Row indices 0..7, stride = 32 → pitch = 32 * 2 = 64 bytes (constant).
    %row_idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dpas}>>
    %row_2d  = tt.expand_dims %row_idx {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dpas}>> -> tensor<8x1xi32, #dpas>
    %stride  = arith.constant dense<32> : tensor<8x1xi32, #dpas>
    %row_off = arith.muli %row_2d, %stride : tensor<8x1xi32, #dpas>
    // Column indices 0..31, unit stride in the column direction.
    %col_idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dpas}>>
    %col_2d  = tt.expand_dims %col_idx {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dpas}>> -> tensor<1x32xi32, #dpas>
    %row_bc  = tt.broadcast %row_off : tensor<8x1xi32, #dpas> -> tensor<8x32xi32, #dpas>
    %col_bc  = tt.broadcast %col_2d  : tensor<1x32xi32, #dpas> -> tensor<8x32xi32, #dpas>
    %offset  = arith.addi %row_bc, %col_bc : tensor<8x32xi32, #dpas>
    %base    = tt.splat %arg0 : !tt.ptr<f16> -> tensor<8x32x!tt.ptr<f16>, #dpas>
    %addr    = tt.addptr %base, %offset : tensor<8x32x!tt.ptr<f16>, #dpas>, tensor<8x32xi32, #dpas>
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<8x32x!tt.ptr<f16>, #dpas>
    tt.return
  }
}
