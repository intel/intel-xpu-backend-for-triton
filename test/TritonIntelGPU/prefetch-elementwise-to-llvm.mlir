// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Test that the lowering pass converts ttig.prefetch on blocked-encoding
// COM: tensors (not DotOperandEncodingAttr) to triton_gen.2DBlockPrefetch
// COM: operations via the cooperative prefetch path.

// COM: Test case: f16 blocked prefetch, 128x64 tensor, 4 warps.
// COM: With 128x64xf16 and 4 warps (256B prefetch support enabled):
// COM:   tileHeight=32, tileWidth=32 -> vBlocks=1, tile_width=32
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

    // CHECK-COUNT-2: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
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

// -----

// COM: Test case: masked cooperative prefetch (epilogue predication from
// COM: the loop pipeliner). The pipeliner wraps the mask in `splat(pred)`,
// COM: and the cooperative path should fold the uniform scalar predicate
// COM: into offsetY = select(pred, 0, baseHeight) so the HW skips the
// COM: prefetch out-of-bounds, mirroring the regular-pointer path's trick.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f16_masked
  tt.func public @prefetch_blocked_f16_masked(%arg0: !tt.ptr<f16>, %pred: i1) {
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

    // COM: tt.getPredMask-shaped mask: splat(%pred).
    %mask = tt.splat %pred : i1 -> tensor<128x64xi1, #blocked>

    // COM: Uniform mask is extracted as a scalar i1 predicate and folded into
    // COM: offsetY via `select(pred, origY, baseHeight)` so the HW skips the
    // COM: prefetch when the predicate is false. Check that the prefetch is
    // COM: still emitted (twice) and that its offsetY comes from an llvm.select.
    // CHECK: %[[SEL0:.*]] = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: triton_gen.2Dblockprefetch %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SEL0]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    // CHECK: %[[SEL1:.*]] = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: triton_gen.2Dblockprefetch %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SEL1]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    ttig.prefetch %ptr, %mask {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>

    tt.return
  }
}

// -----

// COM: Test case: pipeliner-shaped mask `arith.andi(splat(pred), boundsMask)`.
// COM: This is what `tt::getPredMask` produces when the prefetch's underlying
// COM: load already had a mask: the scalar epilogue predicate is splatted and
// COM: AND'd with the original per-element bounds check. The cooperative path
// COM: must honor the uniform `pred` component (folded into offsetY) and drop
// COM: the per-element side (safe for a prefetch — the HW 2D bounds check on
// COM: the tile still rejects out-of-surface offsets).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f16_predmask
  tt.func public @prefetch_blocked_f16_predmask(%arg0: !tt.ptr<f16>, %pred: i1, %bound: tensor<128x64xi32, #blocked>) {
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

    // COM: `tt::getPredMask(..., currentMask=boundsMask, pred=%pred)` layout:
    // COM:   %splat = tt.splat %pred : i1 -> tensor<...xi1>
    // COM:   %mask  = arith.andi %splat, %boundsMask
    %splatPred = tt.splat %pred : i1 -> tensor<128x64xi1, #blocked>
    %boundsMask = arith.cmpi slt, %8, %bound : tensor<128x64xi32, #blocked>
    %mask = arith.andi %splatPred, %boundsMask : tensor<128x64xi1, #blocked>

    // COM: The uniform `%pred` is folded into offsetY via llvm.select; the
    // COM: per-element `%boundsMask` is dropped (safe for a prefetch hint).
    // CHECK: %[[SEL0:.*]] = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: triton_gen.2Dblockprefetch %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SEL0]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    // CHECK: %[[SEL1:.*]] = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
    // CHECK-NEXT: triton_gen.2Dblockprefetch %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SEL1]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    ttig.prefetch %ptr, %mask {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>

    tt.return
  }
}

// -----

// COM: Test case: non-uniform mask on the cooperative path. An `arith.cmpi`
// COM: mask cannot be proven uniform, so the cooperative path must bail. The
// COM: generic pointer path rejects non-dot encodings, so the op is erased
// COM: with a warning and no 2D block prefetch is emitted.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f16_nonuniform_mask
  tt.func public @prefetch_blocked_f16_nonuniform_mask(%arg0: !tt.ptr<f16>, %bound: tensor<128x64xi32, #blocked>) {
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

    // COM: Per-element mask, non-uniform -> cooperative path bails.
    %mask = arith.cmpi slt, %8, %bound : tensor<128x64xi32, #blocked>

    // CHECK-NOT: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr, %mask {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>

    tt.return
  }
}

// -----

// COM: Test case: cooperative prefetch with `cache = 3 : i32` (= CacheModifier::CG).
// COM: Mirrors @prefetch_blocked_f16 but asserts that the CG modifier propagates
// COM: through the prefetch lowering to `cache_control = L1UC_L3C` (L1 bypass).
// COM: This is the path exercised when the `tritonintelgpu-annotate-cache-control`
// COM: pass tags a non-dot streaming load with `.cg` and the pipeliner forwards
// COM: the modifier to the created prefetch — the prefetch must then match the
// COM: load's L1-bypass policy instead of dragging the data back into L1.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_blocked_f16_cg
  tt.func public @prefetch_blocked_f16_cg(%arg0: !tt.ptr<f16>) {
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

    // CHECK-COUNT-2: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 32, v_blocks = 1, cache_control = L1UC_L3C}
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 3 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #blocked>

    tt.return
  }
}

// -----

// COM: Broadcast row (M == 1, row stride 0): the descriptor must use
// COM: base_height = 1 and a non-zero base_width/pitch (see #7267).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_broadcast_row_f32
  tt.func public @prefetch_broadcast_row_f32(%arg0: !tt.ptr<f32>) {
    // COM: 16x32 tensor-of-pointers with a broadcast row (stride 0).
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %3, %2 : tensor<16x32x!tt.ptr<f32>, #blocked>, tensor<16x32xi32, #blocked>

    // COM: base_height comes from the constant 1 (broadcast row); base_width
    // COM: from the contiguous-dim pitch. Tie both captures into the op
    // COM: operands so the check fails if either descriptor value regresses.
    // CHECK: %[[BASE_H_I64:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[BASE_W:.*]] = llvm.trunc %{{.*}} : i64 to i32
    // CHECK: %[[BASE_H:.*]] = llvm.trunc %[[BASE_H_I64]] : i64 to i32
    // CHECK: triton_gen.2Dblockprefetch %{{.*}}, %[[BASE_W]], %[[BASE_H]], %{{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}
    ttig.prefetch %ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<16x32x!tt.ptr<f32>, #blocked>

    tt.return
  }
}
