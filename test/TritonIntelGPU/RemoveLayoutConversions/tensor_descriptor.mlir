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
