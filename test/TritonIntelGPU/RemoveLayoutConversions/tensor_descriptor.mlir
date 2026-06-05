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

#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>

// CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

// COM: ============================================================
// COM: Test 4: Issue #7091 — Expensive descriptor_load rematerialized
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
// COM: Test 5: Negative guard for issue #7091 — When target slice
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
