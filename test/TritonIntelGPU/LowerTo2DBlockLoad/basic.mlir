// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

// COM: Pointer-based load with broadcast (stride=0). The pass traces back
// COM: through broadcast/addptr/splat to find the scalar base pointer (%arg0).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @tensor_of_pointers_load
  tt.func @tensor_of_pointers_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK-DAG: %[[W:.*]] = arith.constant 64 : i32
    // CHECK-DAG: %[[H:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[P:.*]] = arith.constant 64 : i32
    // CHECK-DAG: %[[Z:.*]] = arith.constant 0 : i32
    // CHECK: "ttig.2d_block_load"(%arg0, %[[W]], %[[H]], %[[P]], %[[Z]], %[[Z]])
    // CHECK-NOT: tt.load
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Descriptor-based load with block_io attribute should be converted to
// COM: ttig.2d_block_load with surface params from make_tensor_descriptor.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @tensor_descriptor_load
  tt.func @tensor_descriptor_load(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: tt.descriptor_load
    // CHECK-DAG: %[[ELEM_BYTES:.*]] = arith.constant 2 : i32
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[BW:.*]] = arith.muli %arg2, %[[ELEM_BYTES]]
    // CHECK: %[[ST:.*]] = arith.trunci %arg3
    // CHECK: %[[BP:.*]] = arith.muli %[[ST]], %[[ELEM_BYTES]]
    // CHECK: "ttig.2d_block_load"(%arg0, %[[BW]], %arg1, %[[BP]], %[[ZERO]], %[[ZERO]])
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
