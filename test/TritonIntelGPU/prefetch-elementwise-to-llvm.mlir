// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Test that the lowering pass converts ttig.prefetch on blocked-encoding
// COM: tensors (not DotOperandEncodingAttr) to triton_gen.2DBlockPrefetch
// COM: operations via the cooperative prefetch path.

// COM: Test case: f16 blocked prefetch, 128x64 tensor, 4 warps.
// COM: With 128x64xf16 and 4 warps (256B prefetch support enabled):
// COM:   tileHeight=32, tileWidth=32 -> vBlocks=2, tile_width=16
// COM:   warpsM=2, warpsN=2
// COM:   numTilesPerWarp = (128*64) / (32*32*4) = 2
// COM: Expect 2 triton_gen.2Dblockprefetch ops per prefetch invocation.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f16
  tt.func public @prefetch_blocked_f16(%arg0: !tt.ptr<f16>) {
    // COM: Build a 128x64 tensor-of-pointers with row stride = 64.
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %2 = arith.constant dense<64> : tensor<128x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<128x1xi32, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x64xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %9, %8 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>

    // CHECK-COUNT-2: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, cache_control = L1C_L3C}
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>

    tt.return
  }
}

// -----

// COM: Test case: f32 blocked prefetch, 128x64 tensor, 4 warps.
// COM: With 128x64xf32 and 4 warps (with 256B prefetch support):
// COM:   tileHeight=32, tileWidth=64 (256 bytes / 4 bytes = 64)
// COM:   warpsM=4, warpsN=1
// COM:   numTilesPerWarp = (128*64) / (32*64*4) = 1
// COM: Expect 1 triton_gen.2Dblockprefetch op.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f32
  tt.func public @prefetch_blocked_f32(%arg0: !tt.ptr<f32>) {
    // COM: Build a 128x64 tensor-of-pointers with row stride = 64.
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %2 = arith.constant dense<64> : tensor<128x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<128x1xi32, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x64xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %9, %8 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked>

    // CHECK-COUNT-1: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 32, tile_width = 64, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f32>, #blocked>

    tt.return
  }
}
