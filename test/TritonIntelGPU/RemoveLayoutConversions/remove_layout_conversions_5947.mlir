// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope

// COM: Descriptor equivalent of the original block pointer test for
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/5947

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// COM: Forward propagation removes convert between
// COM: descriptor_load and descriptor_store.
// COM: The descriptor_load anchors layout #blocked1. The intermediate
// COM: convert to #blocked and elementwise op in #blocked are
// COM: rewritten to #blocked1, eliminating the convert_layout ops.

// CHECK-LABEL: @descriptor_load_to_store_propagation
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: %[[NEG:.*]] = arith.negf %[[LOAD]] : tensor<16x64xf32, #[[$BLOCKED]]>
// CHECK: tt.descriptor_store {{.*}}, %[[NEG]] : !tt.tensordesc<16x64xf32>, tensor<16x64xf32, #[[$BLOCKED]]>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_load_to_store_propagation(%desc_in: !tt.tensordesc<16x64xf32>, %desc_out: !tt.tensordesc<16x64xf32>) {
    %c0 = arith.constant 0 : i32
    %load = tt.descriptor_load %desc_in[%c0, %c0] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
    %cvt_to_blocked = ttg.convert_layout %load : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    %neg = arith.negf %cvt_to_blocked : tensor<16x64xf32, #blocked>
    %cvt_back = ttg.convert_layout %neg : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>
    tt.descriptor_store %desc_out[%c0, %c0], %cvt_back : !tt.tensordesc<16x64xf32>, tensor<16x64xf32, #blocked1>
    tt.return
  }
}
