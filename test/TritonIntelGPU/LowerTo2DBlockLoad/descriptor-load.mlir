// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

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

// -----

// COM: Column-major descriptor load. The memory_layout attribute is set to
// COM: column_major, which tells the LLVM lowering to use transposed block
// COM: reads.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_column_major
  tt.func @descriptor_load_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<32x64xf16, #dot1> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: tt.descriptor_load
    // CHECK: "ttig.2d_block_load"
    // CHECK-SAME: memory_layout = 1
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x32xf16> -> tensor<32x64xf16, #dot1>
    tt.return %0 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: Descriptor load with NaN padding. The pass should create a NaN splat
// COM: as 'other' and leave mask absent.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pad_nan
  tt.func @descriptor_load_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
    // CHECK: %[[NAN:.*]] = arith.constant 0x7E00 : f16
    // CHECK: %[[NAN_SPLAT:.*]] = tt.splat %[[NAN]]
    // CHECK: "ttig.2d_block_load"
    // CHECK-NOT: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
