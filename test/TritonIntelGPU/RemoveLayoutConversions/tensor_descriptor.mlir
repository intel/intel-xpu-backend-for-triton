// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// Test that tt.descriptor_load/tt.descriptor_store are treated as layout
// anchors and that unnecessary convert_layout ops are eliminated when tensor
// descriptors are used.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 1: Forward propagation from descriptor_load anchor.
// COM: descriptor_load produces #blocked1; the elementwise arith.addf
// COM: inherits #blocked1 and the convert_layout to #blocked1 is eliminated.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_forward_propagation
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[LOAD]] : tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: tt.return %[[ADD]] : tensor<16x64xf32, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_forward_propagation(%desc: !tt.tensordesc<tensor<16x64xf32>>) -> tensor<16x64xf32, #blocked1> {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<tensor<16x64xf32>> -> tensor<16x64xf32, #blocked1>
    %add = arith.addf %load, %load : tensor<16x64xf32, #blocked1>
    %cvt = ttg.convert_layout %add : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked1>
    tt.return %cvt : tensor<16x64xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 2: Forward propagation removes convert between
// COM: descriptor_load and descriptor_store.
// COM: The descriptor_load anchors layout #blocked1, an elementwise
// COM: op in #blocked is converted, and the convert_layout feeding
// COM: descriptor_store is eliminated.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_to_store_propagation
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: %[[NEG:.*]] = arith.negf %[[LOAD]] : tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: tt.descriptor_store {{.*}}, %[[NEG]] : !tt.tensordesc<tensor<16x64xf32>>, tensor<16x64xf32, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_to_store_propagation(%desc_in: !tt.tensordesc<tensor<16x64xf32>>, %desc_out: !tt.tensordesc<tensor<16x64xf32>>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc_in[%c0, %c0] : !tt.tensordesc<tensor<16x64xf32>> -> tensor<16x64xf32, #blocked1>
    %neg = arith.negf %load : tensor<16x64xf32, #blocked1>
    %cvt = ttg.convert_layout %neg : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked1>
    tt.descriptor_store %desc_out[%c0, %c0], %cvt : !tt.tensordesc<tensor<16x64xf32>>, tensor<16x64xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 3: Descriptor store as anchor — the store's encoding
// COM: propagates backward through the cheap splat, eliminating
// COM: the convert_layout entirely.
// COM: ============================================================

// CHECK-LABEL: @descriptor_store_propagates_backward
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[SPLAT:.*]] = tt.splat {{.*}} : f32 -> tensor<16x64xf32, #[[$BLOCKED1]]>
// CHECK: tt.descriptor_store {{.*}}, %[[SPLAT]] : !tt.tensordesc<tensor<16x64xf32>>, tensor<16x64xf32, #[[$BLOCKED1]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_store_propagates_backward(%desc: !tt.tensordesc<tensor<16x64xf32>>, %val: f32) {
    %c0 = arith.constant 0 : i32
    %splat = tt.splat %val : f32 -> tensor<16x64xf32, #blocked>
    %cvt = ttg.convert_layout %splat : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>
    tt.descriptor_store %desc[%c0, %c0], %cvt : !tt.tensordesc<tensor<16x64xf32>>, tensor<16x64xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: ============================================================
// COM: Test 4: descriptor_load is not rematerialized during backward
// COM: pass. The expensive load op should not be duplicated.
// COM: The convert_layout between two different true layouts is
// COM: preserved; the descriptor_load is NOT cloned with the new
// COM: encoding.
// COM: ============================================================

// CHECK-LABEL: @descriptor_load_not_rematerialized
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED1]]>
// CHECK: ttg.convert_layout %[[LOAD]]
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_not_rematerialized(%desc: !tt.tensordesc<tensor<16x64xf32>>, %out: tensor<16x64x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<tensor<16x64xf32>> -> tensor<16x64xf32, #blocked1>
    %cvt = ttg.convert_layout %load : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    tt.store %out, %cvt : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
