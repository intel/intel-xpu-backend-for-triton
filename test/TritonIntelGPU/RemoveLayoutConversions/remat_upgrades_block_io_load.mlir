// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: A block_io load whose coalesced layout does not validate as a 2D block load is
// COM: anchored by isExpensiveLoadOrStore, so it is normally not rematerializable (a
// COM: relabel usually de-coalesces it into a worse gather, issue #7090). When the layout
// COM: the rematerialization would assign DOES validate as a 2D block load, the relabel
// COM: upgrades the load instead, and additionally removes the convert_layout (an SLM
// COM: round trip with barriers). rematUpgradesExpensiveLoad allows exactly that case.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "pvc"} {
  // COM: GEMM epilogue post-op: the addend is added to a DPAS accumulator, so the only
  // COM: layout on its forward path is the bare #ttig.dpas accumulator encoding (never a
  // COM: dot-operand encoding). The addend load must take that layout directly.
  // CHECK-LABEL: @dpas_accumulator_postop_addend
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %{{.*}}[{{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #[[$MMA:.+]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[LOAD]] : tensor<256x256xf32, #[[$MMA]]>
  // CHECK: tt.descriptor_store %{{.*}}[{{.*}}], %[[ADD]] {{.*}} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #[[$MMA]]>
  tt.func public @dpas_accumulator_postop_addend(%d_desc: !tt.tensordesc<256x256xf32>, %c_desc: !tt.tensordesc<256x256xf32>, %acc: tensor<256x256xf32, #mma>) {
    %c0_i32 = arith.constant 0 : i32
    %d = tt.descriptor_load %d_desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #blocked>
    %d_cvt = ttg.convert_layout %d : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #mma>
    %c = arith.addf %acc, %d_cvt : tensor<256x256xf32, #mma>
    tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %c {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

// COM: Negative case: the convert target does NOT validate as a 2D block load (lanes
// COM: advance along the non-contiguous dimension, i.e. a de-coalescing relabel). The
// COM: anchor must hold: the load keeps its coalesced layout and the convert is preserved.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "pvc"} {
  // CHECK-LABEL: @decoalescing_target_keeps_anchor
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %{{.*}}[{{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #[[$LOAD:.+]]>
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<256x256xf32, #[[$LOAD]]> -> tensor<256x256xf32, #[[$STORE:.+]]>
  // CHECK: tt.descriptor_store %{{.*}}[{{.*}}], %[[CVT]] {{.*}} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #[[$STORE]]>
  tt.func public @decoalescing_target_keeps_anchor(%d_desc: !tt.tensordesc<256x256xf32>, %c_desc: !tt.tensordesc<256x256xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %d = tt.descriptor_load %d_desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #blocked>
    %d_cvt = ttg.convert_layout %d : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #blocked1>
    tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %d_cvt {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #blocked1>
    tt.return
  }
}

// -----

// COM: Mixed-use load: one convert target validates as a 2D block load (#ttig.dpas
// COM: accumulator) and one does not (de-coalescing store layout). The decision is made
// COM: per target, so the validating path gets a relabeled load while the anchor still
// COM: holds for the other path -- no single "is this load cheap" verdict can be wrong
// COM: for one of the two users. The load is duplicated (one per layout), which is the
// COM: same trade the cost model already makes for a non-anchored load: one extra 2D
// COM: block load in exchange for removing a convert_layout (an SLM round trip).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "pvc"} {
  // CHECK-LABEL: @mixed_use_decides_per_target
  // COM: The validating (#ttig.dpas) path gets its own load in the accumulator layout.
  // CHECK: %[[LOAD_MMA:.*]] = tt.descriptor_load %[[DESC:.*]][{{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #[[$MMA:.+]]>
  // COM: The de-coalescing path keeps a load in the coalesced layout -- it is NOT dragged
  // COM: into the gather layout by the other user's relabel (issue #7090 regression class).
  // CHECK: %[[LOAD_COAL:.*]] = tt.descriptor_load %[[DESC]][{{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #[[$COAL:.+]]>
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[LOAD_MMA]] : tensor<256x256xf32, #[[$MMA]]>
  // CHECK: tt.descriptor_store %{{.*}}[{{.*}}], %[[ADD]] {{.*}} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #[[$MMA]]>
  // COM: ... and that path's convert_layout is preserved, so the anchor still holds there.
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[LOAD_COAL]] : tensor<256x256xf32, #[[$COAL]]> -> tensor<256x256xf32, #[[$STORE:.+]]>
  // CHECK: tt.descriptor_store %{{.*}}[{{.*}}], %[[CVT]] {{.*}} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #[[$STORE]]>
  tt.func public @mixed_use_decides_per_target(%d_desc: !tt.tensordesc<256x256xf32>, %c_desc: !tt.tensordesc<256x256xf32>, %e_desc: !tt.tensordesc<256x256xf32>, %acc: tensor<256x256xf32, #mma>) {
    %c0_i32 = arith.constant 0 : i32
    %d = tt.descriptor_load %d_desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32> -> tensor<256x256xf32, #blocked>
    %d_mma = ttg.convert_layout %d : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #mma>
    %c = arith.addf %acc, %d_mma : tensor<256x256xf32, #mma>
    tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %c {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #mma>
    %d_bad = ttg.convert_layout %d : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #blocked1>
    tt.descriptor_store %e_desc[%c0_i32, %c0_i32], %d_bad {ttig.block_io = "row_major"} : !tt.tensordesc<256x256xf32>, tensor<256x256xf32, #blocked1>
    tt.return
  }
}
