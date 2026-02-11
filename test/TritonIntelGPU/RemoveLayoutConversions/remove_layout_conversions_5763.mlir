// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16], order = [1, 0]}>
// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 8], order = [1, 0]}>
module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @remove_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<2048> : tensor<1x2048xi32, #blocked>
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2048xi32, #blocked>
    %2 = arith.cmpi slt, %1, %cst : tensor<1x2048xi32, #blocked>
    // CHECK: tt.assert {{.*}} : tensor<1x2048xi1, #[[$BLOCKED]]>
    tt.assert %2, "assert text" : tensor<1x2048xi1, #blocked>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x2048x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %1 : tensor<1x2048x!tt.ptr<f16>, #blocked>, tensor<1x2048xi32, #blocked>
    // COM: the following conversion should be removed because the layout used
    // COM: by the load operation is backward propagated, and the resulting
    // COM: layout should be used by tt.assert.
    %5 = ttg.convert_layout %4 : tensor<1x2048x!tt.ptr<f16>, #blocked> -> tensor<1x2048x!tt.ptr<f16>, #blocked1>
    %6 = tt.load %5 evictionPolicy = evict_last : tensor<1x2048x!tt.ptr<f16>, #blocked1>
    tt.print " " {hex = false, isSigned = array<i32: 0>} : %6 : tensor<1x2048xf16, #blocked1>
    tt.return
  }
}
