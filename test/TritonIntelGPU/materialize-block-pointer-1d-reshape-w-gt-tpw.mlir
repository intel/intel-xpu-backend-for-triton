// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file \
// RUN:   -tritonintelgpu-materialize-block-pointer \
// RUN:   --tritonintelgpu-lower-to-2d-block-load \
// RUN:   --intel-allocate-shared-memory \
// RUN:   --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Regression tests for the W > threadsPerWarp correctness bug in the 1D→2D
// reshape paths.
//
// Root cause: the hand-built BlockIOTileSizeInfo in LoadStoreOpToLLVM.cpp
// hardcoded numElemPerPackedVal=1 / tileWidth=W. When W > threadsPerWarp,
// each lane owns W/tpw adjacent columns in the encoding; the hardware delivers
// packed values at intervals of threadsPerWarp, so the tile must pack those
// adjacent columns into a wider element type. The shared getBlockIOTileSize
// helper does this via packRegister; the hand-built path did not.
//
// For i8 / W=64 / tpw=32, the correct tile packs two adjacent i8 into one i16:
//   WRONG: elem_size_in_bits=8,  tile_width=64  (lane l gets cols l and l+32)
//   RIGHT: elem_size_in_bits=16, tile_width=32  (lane l gets cols 2l,2l+1 as i16)
//
// The tests below were FAILING before the fix (wrong elem_size/tile_width
// asserted) and PASSING after it.

// ============================================================
// Test 1: STORE i8, W=64 > tpw=32, H=1, BMG.
//
// matchStridedPattern: W=64 clears check2DBlockAddressPayloadRestriction(8,64).
// reshape1DStridedStore: H=1 ≤ maxPerWarpHeight=8, numWarps=1.
// Reshaped encoding: sizePerThread=[1,2], threadsPerWarp=[1,32].
// Expected: two adjacent i8 packed → elem_size=16, tile_width=32.
// ============================================================

// CHECK-LABEL: llvm.func spir_kernelcc @store_i8_w64_h1_tpw32
// CHECK:       triton_gen.2Dblockstore
// CHECK-SAME:      {elem_size_in_bits = 16, tile_width = 32, tile_height = 1
// CHECK-NOT:   {elem_size_in_bits = 8, tile_width = 64

#blocked1d_t1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @store_i8_w64_h1_tpw32(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                                   %arg1: tensor<64xi8, #blocked1d_t1>) {
    %idx = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked1d_t1>
    %cW  = arith.constant dense<64>  : tensor<64xi32, #blocked1d_t1>
    %cS  = arith.constant dense<128> : tensor<64xi32, #blocked1d_t1>
    %r   = arith.remui %idx, %cW : tensor<64xi32, #blocked1d_t1>
    %d   = arith.divui %idx, %cW : tensor<64xi32, #blocked1d_t1>
    %m   = arith.muli  %d,   %cS : tensor<64xi32, #blocked1d_t1>
    %off = arith.addi  %r,   %m  : tensor<64xi32, #blocked1d_t1>
    %bp  = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #blocked1d_t1>
    %p   = tt.addptr %bp, %off : tensor<64x!tt.ptr<i8>, #blocked1d_t1>, tensor<64xi32, #blocked1d_t1>
    tt.store %p, %arg1 : tensor<64x!tt.ptr<i8>, #blocked1d_t1>
    tt.return
  }
}

// -----

// ============================================================
// Test 2: STORE f16, W=32 > tpw=16, H=1, PVC (16-wide subgroup).
//
// Same tested shape as test_1d_reshape_h1_blockstore (W=32, f16) but on a
// 16-wide subgroup. W=32 > tpw=16, so sizePerThread=[1,2] again.
// Expected: two adjacent f16 packed → elem_size=32, tile_width=16.
// ============================================================

// CHECK-LABEL: llvm.func spir_kernelcc @store_f16_w32_h1_tpw16
// CHECK:       triton_gen.2Dblockstore
// CHECK-SAME:      {elem_size_in_bits = 32, tile_width = 16, tile_height = 1
// CHECK-NOT:   {elem_size_in_bits = 16, tile_width = 32, tile_height = 1

#blocked1d_t2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @store_f16_w32_h1_tpw16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                    %arg1: tensor<32xf16, #blocked1d_t2>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d_t2>
    %cW  = arith.constant dense<32> : tensor<32xi32, #blocked1d_t2>
    %cS  = arith.constant dense<96> : tensor<32xi32, #blocked1d_t2>
    %r   = arith.remui %idx, %cW : tensor<32xi32, #blocked1d_t2>
    %d   = arith.divui %idx, %cW : tensor<32xi32, #blocked1d_t2>
    %m   = arith.muli  %d,   %cS : tensor<32xi32, #blocked1d_t2>
    %off = arith.addi  %r,   %m  : tensor<32xi32, #blocked1d_t2>
    %bp  = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d_t2>
    %p   = tt.addptr %bp, %off : tensor<32x!tt.ptr<f16>, #blocked1d_t2>, tensor<32xi32, #blocked1d_t2>
    tt.store %p, %arg1 : tensor<32x!tt.ptr<f16>, #blocked1d_t2>
    tt.return
  }
}

// -----

// ============================================================
// Test 3: LOAD i8, W=64 > tpw=32, H=8, BMG.
//
// The load path is NOT gated on H=1 (maxPerWarpHeight=32), so this tests
// the more common multi-row case. reshape1DStridedLoad builds
// loadEnc = sizePerThread=[8,2], threadsPerWarp=[1,32].
// Expected: elem_size=16, tile_width=32, tile_height=8.
// ============================================================

// CHECK-LABEL: llvm.func spir_kernelcc @load_i8_w64_h8_tpw32
// CHECK:       triton_gen.2Dblockload
// CHECK-SAME:      {elem_size_in_bits = 16, tile_width = 32, tile_height = 8
// CHECK-NOT:   {elem_size_in_bits = 8, tile_width = 64

#blocked1d_t3 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @load_i8_w64_h8_tpw32(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32})
      -> tensor<512xi8, #blocked1d_t3> {
    %idx = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked1d_t3>
    %cW  = arith.constant dense<64>  : tensor<512xi32, #blocked1d_t3>
    %cS  = arith.constant dense<128> : tensor<512xi32, #blocked1d_t3>
    %r   = arith.remui %idx, %cW : tensor<512xi32, #blocked1d_t3>
    %d   = arith.divui %idx, %cW : tensor<512xi32, #blocked1d_t3>
    %m   = arith.muli  %d,   %cS : tensor<512xi32, #blocked1d_t3>
    %off = arith.addi  %r,   %m  : tensor<512xi32, #blocked1d_t3>
    %bp  = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked1d_t3>
    %p   = tt.addptr %bp, %off : tensor<512x!tt.ptr<i8>, #blocked1d_t3>, tensor<512xi32, #blocked1d_t3>
    %v   = tt.load %p : tensor<512x!tt.ptr<i8>, #blocked1d_t3>
    tt.return %v : tensor<512xi8, #blocked1d_t3>
  }
}

// -----

// ============================================================
// Control 1: STORE i8, W=32 == tpw=32, H=1 — existing passing case.
// Should still emit elem_size=8, tile_width=32 (no packing needed).
// ============================================================

// CHECK-LABEL: llvm.func spir_kernelcc @store_i8_w32_h1_tpw32_control
// CHECK:       triton_gen.2Dblockstore
// CHECK-SAME:      {elem_size_in_bits = 8, tile_width = 32, tile_height = 1

#blocked1d_c1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @store_i8_w32_h1_tpw32_control(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                                           %arg1: tensor<32xi8, #blocked1d_c1>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d_c1>
    %cW  = arith.constant dense<32> : tensor<32xi32, #blocked1d_c1>
    %cS  = arith.constant dense<128> : tensor<32xi32, #blocked1d_c1>
    %r   = arith.remui %idx, %cW : tensor<32xi32, #blocked1d_c1>
    %d   = arith.divui %idx, %cW : tensor<32xi32, #blocked1d_c1>
    %m   = arith.muli  %d,   %cS : tensor<32xi32, #blocked1d_c1>
    %off = arith.addi  %r,   %m  : tensor<32xi32, #blocked1d_c1>
    %bp  = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x!tt.ptr<i8>, #blocked1d_c1>
    %p   = tt.addptr %bp, %off : tensor<32x!tt.ptr<i8>, #blocked1d_c1>, tensor<32xi32, #blocked1d_c1>
    tt.store %p, %arg1 : tensor<32x!tt.ptr<i8>, #blocked1d_c1>
    tt.return
  }
}

// -----

// ============================================================
// Control 2: LOAD f16, W=32 == tpw=32, H=32 — existing passing case.
// Should still emit elem_size=16, tile_width=32, tile_height=8.
// ============================================================

// CHECK-LABEL: llvm.func spir_kernelcc @load_f16_w32_h32_tpw32_control
// CHECK:       triton_gen.2Dblockload
// CHECK-SAME:      {elem_size_in_bits = 16, tile_width = 32, tile_height = 8

#blocked1d_c2 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @load_f16_w32_h32_tpw32_control(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32})
      -> tensor<1024xf16, #blocked1d_c2> {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d_c2>
    %cW  = arith.constant dense<32> : tensor<1024xi32, #blocked1d_c2>
    %cS  = arith.constant dense<96> : tensor<1024xi32, #blocked1d_c2>
    %r   = arith.remui %idx, %cW : tensor<1024xi32, #blocked1d_c2>
    %d   = arith.divui %idx, %cW : tensor<1024xi32, #blocked1d_c2>
    %m   = arith.muli  %d,   %cS : tensor<1024xi32, #blocked1d_c2>
    %off = arith.addi  %r,   %m  : tensor<1024xi32, #blocked1d_c2>
    %bp  = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d_c2>
    %p   = tt.addptr %bp, %off : tensor<1024x!tt.ptr<f16>, #blocked1d_c2>, tensor<1024xi32, #blocked1d_c2>
    %v   = tt.load %p : tensor<1024x!tt.ptr<f16>, #blocked1d_c2>
    tt.return %v : tensor<1024xf16, #blocked1d_c2>
  }
}
