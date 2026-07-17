// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope

// Test that tt.descriptor_load/tt.descriptor_store are treated as layout
// anchors and that unnecessary convert_layout ops are eliminated when tensor
// descriptors are used.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 1: Forward propagation from descriptor_load anchor.
// COM: descriptor_load produces #blocked1; the convert_layout to
// COM: #blocked and the elementwise arith.addf in #blocked are
// COM: rewritten to #blocked1, eliminating both convert_layout ops.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_forward_propagation
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[LOAD]] : tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: tt.return %[[ADD]] : tensor<16x64xf32, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_forward_propagation(%desc: !tt.tensordesc<16x64xf32>) -> tensor<16x64xf32, #blocked1> {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
    %cvt_to_blocked = ttg.convert_layout %load : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    %add = arith.addf %cvt_to_blocked, %cvt_to_blocked : tensor<16x64xf32, #blocked>
    %cvt_back = ttg.convert_layout %add : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>
    tt.return %cvt_back : tensor<16x64xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 2: descriptor_load is not rematerialized during backward
// COM: pass. The expensive load op should not be duplicated.
// COM: The convert_layout between two different true layouts is
// COM: preserved; the descriptor_load is NOT cloned with the new
// COM: encoding.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_not_rematerialized
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED1]]>
// CHECK: ttg.convert_layout %[[LOAD]] : tensor<16x64xf32, #[[$BLOCKED1]]> -> tensor<16x64xf32, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_not_rematerialized(%desc: !tt.tensordesc<16x64xf32>, %out: tensor<16x64x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
    %cvt = ttg.convert_layout %load : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    tt.store %out, %cvt : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

// COM: ============================================================
// COM: Test 3: Descriptor store drops a trailing convert_layout from
// COM: a DPAS source layout and stores directly from the source.
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_dpas_source_forwarding
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.descriptor_store {{.*}}, %arg1 : !tt.tensordesc<8x32xf16>, tensor<8x32xf16, #mma>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_dpas_source_forwarding(%arg0: !tt.tensordesc<8x32xf16>, %arg1: tensor<8x32xf16, #mma>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.convert_layout %arg1 : tensor<8x32xf16, #mma> -> tensor<8x32xf16, #blocked>
    tt.descriptor_store %arg0[%c0_i32, %c0_i32], %0 : !tt.tensordesc<8x32xf16>, tensor<8x32xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 4: Issue #7080 — 2D block I/O cost model prevents
// COM: degradation. Descriptor_load in a better block-io layout
// COM: (higher vectorization, sizePerThread=[1,4]) is NOT
// COM: rematerialized into a worse layout (scatter-like [1,1])
// COM: even though backward remat would eliminate the convert.
// COM: The convert_layout is PRESERVED to avoid de-vectorizing
// COM: the hardware 2D block load.
// COM: ============================================================

// CHECK-LABEL: @block_io_descriptor_load_not_degraded
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<32x32xf16, #[[$BLOCKED1]]>
// CHECK: ttg.convert_layout %[[LOAD]] : tensor<32x32xf16, #[[$BLOCKED1]]> -> tensor<32x32xf16, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @block_io_descriptor_load_not_degraded(%desc: !tt.tensordesc<32x32xf16>, %out: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16> -> tensor<32x32xf16, #blocked1>
    %cvt = ttg.convert_layout %load : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked>
    tt.store %out, %cvt : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [1, 0]}>

// CHECK: #[[$BLOCKED3:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [1, 0]}>

// COM: ============================================================
// COM: Test 5: Issue #7080 — 2D block I/O cost model ALLOWS an
// COM: improving remat. The source layout (#blocked1, warpsPerCTA
// COM: [4,2]) yields a 2D block tile of height 2 (16 messages for a
// COM: 32x32 tile); the target layout (#blocked3, warpsPerCTA [1,8])
// COM: yields tile height 32 (4 messages) — strictly cheaper. The
// COM: cost model permits the remat: the convert_layout is
// COM: ELIMINATED and the descriptor_load directly produces the
// COM: better encoding. This is the case a blunt "block_io load ->
// COM: blocked target = reject" guard would wrongly forbid.
// COM: ============================================================

// CHECK-LABEL: @block_io_descriptor_load_improved
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<32x32xf16, #[[$BLOCKED3]]>
// CHECK: tt.store {{.*}}, %[[LOAD]] : tensor<32x32x!tt.ptr<f16>, #[[$BLOCKED3]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @block_io_descriptor_load_improved(%desc: !tt.tensordesc<32x32xf16>, %out: tensor<32x32x!tt.ptr<f16>, #blocked3>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16> -> tensor<32x32xf16, #blocked1>
    %cvt = ttg.convert_layout %load : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked3>
    tt.store %out, %cvt : tensor<32x32x!tt.ptr<f16>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

// COM: ============================================================
// COM: Test 6: Descriptor store PRESERVES an order-changing (transpose)
// COM: convert_layout. Coalesce inserts a [0,1] -> [1,0] convert so the
// COM: store is row-major / coalesced; RLC must NOT fold it away (doing
// COM: so demotes the store to a scalar scatter). GitHub issue #7093.
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_transpose_convert_preserved
// CHECK: %[[CVT:.*]] = ttg.convert_layout {{.*}} -> tensor<16x64xf16, #blocked1>
// CHECK: tt.descriptor_store {{.*}}, %[[CVT]] : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked1>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_transpose_convert_preserved(%desc: !tt.tensordesc<16x64xf16>, %src: tensor<16x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %cvt = ttg.convert_layout %src : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #blocked1>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

// COM: ============================================================
// COM: Test 7: Descriptor store still FOLDS a same-order convert_layout
// COM: ([1,0] -> [1,0]). Such a convert is only a re-tiling, not a
// COM: transpose, so it must not be anchored (cf. issue #4866).
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_same_order_convert_folded
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.descriptor_store {{.*}}, %arg1 : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_same_order_convert_folded(%desc: !tt.tensordesc<16x64xf16>, %src: tensor<16x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %cvt = ttg.convert_layout %src : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #blocked1>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

// COM: ============================================================
// COM: Test 8: Descriptor store PRESERVES a SAME-order convert whose
// COM: lane distribution is transposed (threadsPerWarp [16,1] -> [1,16],
// COM: both order = [1,0]). The `order` attribute is identical on both
// COM: sides, so an order-based check would wrongly fold; the lane-fast
// COM: dim differs (0 -> 1), so the convert is a genuine within-subgroup
// COM: transpose that must be anchored (else the row-major store demotes
// COM: to a scalar scatter). GitHub issue #7093 (same-order gap).
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_same_order_lane_transpose_preserved
// CHECK: %[[CVT:.*]] = ttg.convert_layout {{.*}} -> tensor<16x64xf16, #blocked1>
// CHECK: tt.descriptor_store {{.*}}, %[[CVT]] : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked1>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_same_order_lane_transpose_preserved(%desc: !tt.tensordesc<16x64xf16>, %src: tensor<16x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %cvt = ttg.convert_layout %src : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #blocked1>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<16x64xf16>, tensor<16x64xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>

// CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

// COM: ============================================================
// COM: Test 7: Issue #7091 — Expensive descriptor_load rematerialized
// COM: into slice layout when sole consumer is convert_layout feeding
// COM: single-use expand_dims/trans/broadcast chain (non-degenerate
// COM: target). The convert_layout is eliminated; descriptor_load
// COM: directly produces the slice encoding.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_rematerialized_into_slice
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<1024xf16, #ttg.slice<{dim = 1, parent = #[[$BLOCKED1]]}>>
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[LOAD]] {axis = 1 : i32} : tensor<1024xf16, #ttg.slice<{dim = 1, parent = #[[$BLOCKED1]]}>> -> tensor<1024x1xf16, #[[$BLOCKED1]]>
// CHECK: %[[TRANS:.*]] = tt.trans %[[EXPAND]] {{.*}} -> tensor<1x1024xf16, #[[$BLOCKED]]>
// CHECK: %[[BCAST:.*]] = tt.broadcast %[[TRANS]] : tensor<1x1024xf16, #[[$BLOCKED]]> -> tensor<4x1024xf16, #[[$BLOCKED]]>
// CHECK: tt.return %[[BCAST]]
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_load_rematerialized_into_slice(%desc: !tt.tensordesc<1024xf16>) -> tensor<4x1024xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %0 = tt.descriptor_load %desc[%c0] : !tt.tensordesc<1024xf16> -> tensor<1024xf16, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1024xf16, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<1024x1xf16, #blocked2>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<1024x1xf16, #blocked2> -> tensor<1x1024xf16, #blocked>
    %4 = tt.broadcast %3 : tensor<1x1024xf16, #blocked> -> tensor<4x1024xf16, #blocked>
    tt.return %4 : tensor<4x1024xf16, #blocked>
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>

// CHECK-DAG: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// CHECK-DAG: #[[$BLOCKED2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>

// COM: ============================================================
// COM: Test 8: Negative guard for issue #7091 — When target slice
// COM: parent layout is degenerate (sizePerThread all-ones),
// COM: descriptor_load is NOT rematerialized. The convert_layout
// COM: is preserved to prevent layout-fixpoint regressions.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_not_rematerialized_degenerate
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<1024xf16, #[[$BLOCKED1]]>
// CHECK: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<1024xf16, #[[$BLOCKED1]]> -> tensor<1024xf16, #ttg.slice<{dim = 1, parent = #[[$BLOCKED2]]}>>
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[CVT]] {axis = 1 : i32} : tensor<1024xf16, #ttg.slice<{dim = 1, parent = #[[$BLOCKED2]]}>> -> tensor<1024x1xf16, #[[$BLOCKED2]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_load_not_rematerialized_degenerate(%desc: !tt.tensordesc<1024xf16>) -> tensor<4x1024xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %0 = tt.descriptor_load %desc[%c0] : !tt.tensordesc<1024xf16> -> tensor<1024xf16, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1024xf16, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<1024x1xf16, #blocked2>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<1024x1xf16, #blocked2> -> tensor<1x1024xf16, #blocked>
    %4 = tt.broadcast %3 : tensor<1x1024xf16, #blocked> -> tensor<4x1024xf16, #blocked>
    tt.return %4 : tensor<4x1024xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>

// COM: ============================================================
// COM: Test 9: Descriptor store PRESERVES a SAME-order, SAME-lane-fast-dim
// COM: convert when its dst is the canonical coalesced layout and the chain
// COM: root is not a dot. Coalesce inserts the convert to the coalesced
// COM: layout (warpsPerCTA [1,8] along the contiguous dim); forward
// COM: propagation from the producer's compute layout (warpsPerCTA [2,4])
// COM: would otherwise fold it and demote global store coalescing. Both
// COM: layouts share order [1,0] and lane-fast dim 1, so the #7093
// COM: lane-transpose guard does not fire; the canonical-layout guard does.
// COM: GitHub issue #7104.
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_canonical_coalesced_preserved
// CHECK: %[[CVT:.*]] = ttg.convert_layout {{.*}} -> tensor<2x1024xf32, #blocked1>
// CHECK: tt.descriptor_store {{.*}}, %[[CVT]] : !tt.tensordesc<2x1024xf32>, tensor<2x1024xf32, #blocked1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_canonical_coalesced_preserved(%desc: !tt.tensordesc<2x1024xf32>, %src: tensor<2x1024xf32, #blocked>) {
    %c0 = arith.constant 0 : i32
    %cvt = ttg.convert_layout %src : tensor<2x1024xf32, #blocked> -> tensor<2x1024xf32, #blocked1>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<2x1024xf32>, tensor<2x1024xf32, #blocked1>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
// Canonical coalesced layout tritongpu-coalesce assigns to a 64x64xf16
// descriptor-store operand at num-warps=8, threads-per-warp=16.
#coalesced = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

// COM: ============================================================
// COM: Test 10: Descriptor store FOLDS a convert from a dot result even when
// COM: the convert's dst IS the canonical coalesced layout. When the
// COM: convert-chain root is a tt.dot, rootIsDot=true suppresses the
// COM: canonical-coalesced anchor (issue #4866): a dot result re-tiled for the
// COM: store SHOULD take the dot (DPAS) layout, so the convert must fold. This
// COM: is the negative counterpart of Test 9 -- same dst-is-canonical
// COM: condition, but a dot root flips the anchor decision from preserve to
// COM: fold. GitHub issue #7104 / #4866.
// COM: ============================================================

// CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}>
// CHECK-LABEL: @descriptor_store_dot_root_convert_folded
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<64x64xf16>, tensor<64x64xf16, #[[$DPAS]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_dot_root_convert_folded(
      %desc: !tt.tensordesc<64x64xf16>,
      %a: tensor<64x64xf16, #dot0>,
      %b: tensor<64x64xf16, #dot1>,
      %acc: tensor<64x64xf16, #dpas>) {
    %c0 = arith.constant 0 : i32
    %dot = tt.dot %a, %b, %acc, inputPrecision = ieee :
        tensor<64x64xf16, #dot0> * tensor<64x64xf16, #dot1> -> tensor<64x64xf16, #dpas>
    %cvt = ttg.convert_layout %dot : tensor<64x64xf16, #dpas> -> tensor<64x64xf16, #coalesced>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<64x64xf16>, tensor<64x64xf16, #coalesced>
    tt.return
  }
}
