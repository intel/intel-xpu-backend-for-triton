// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Test that the cost gate in hoistConvertOnTopOfExtOrBroadcast prevents
// COM: unprofitable convert hoisting. The convert_layout operates on a small
// COM: tensor<32xf32> after two reductions. If it were hoisted above the
// COM: broadcast, it would operate on the larger tensor<1x32x32xf32> input,
// COM: making the conversion more expensive. The isRematBeneficial check
// COM: detects this and keeps the convert in its original position.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_convert_unprofitable
  // CHECK: tt.broadcast
  // CHECK: tt.reduce
  // CHECK: tt.reduce
  // CHECK: ttg.convert_layout
  tt.func public @hoist_convert_unprofitable(%arg0: tensor<1x32x32xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #blocked1}>}>> {
    %0 = tt.broadcast %arg0 : tensor<1x32x32xf32, #blocked> -> tensor<32x32x32xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %4 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<32x32x32xf32, #blocked>) -> tensor<32x32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %4 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #blocked}>}>>
    %3 = ttg.convert_layout %2 : tensor<32xf32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #blocked}>}>> -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #blocked1}>}>>
    tt.return %3 : tensor<32xf32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 0, parent = #blocked1}>}>>
  }
}
