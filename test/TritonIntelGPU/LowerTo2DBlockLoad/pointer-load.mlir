// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

// COM: Pointer-based load with broadcast (stride=0). The pass converts this
// COM: to ttig.2d_block_ptr_load, retaining the full pointer tensor.
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
    // CHECK: "ttig.2d_block_ptr_load"(%4, %[[W]], %[[H]], %[[P]])
    // CHECK-NOT: tt.load
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Masked pointer load with both mask and other. Both are forwarded
// COM: to the ttig.2d_block_ptr_load op.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_with_other
  tt.func @masked_pointer_load_with_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %5 = tt.splat %arg1 : i32 -> tensor<1x32xi32, #dot0>
    %6 = arith.cmpi slt, %5, %1 : tensor<1x32xi32, #dot0>
    %mask = tt.broadcast %6 : tensor<1x32xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK: "ttig.2d_block_ptr_load"
    // CHECK-NOT: tt.load
    %7 = tt.load %4, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Masked pointer load WITHOUT an explicit 'other' value. The pass must
// COM: synthesize a zero splat so the verifier constraint (other required when
// COM: mask is present) is satisfied.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_no_other
  tt.func @masked_pointer_load_no_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %5 = tt.splat %arg1 : i32 -> tensor<1x32xi32, #dot0>
    %6 = arith.cmpi slt, %5, %1 : tensor<1x32xi32, #dot0>
    %mask = tt.broadcast %6 : tensor<1x32xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK: %[[ZERO_CST:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[ZERO_SPLAT:.*]] = tt.splat %[[ZERO_CST]]
    // CHECK: "ttig.2d_block_ptr_load"
    // CHECK-NOT: tt.load
    %7 = tt.load %4, %mask {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}
