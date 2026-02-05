// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// CHECK-NOT: ttg.convert_layout
// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 8], order = [1, 0]}>
// CHECK-NOT: #ttg.blocked
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 8], order = [1, 0]}>
module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @remove_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<true> : tensor<1x2048xi1, #blocked>
    %cst_0 = arith.constant dense<2048> : tensor<1x2048xi32, #blocked>
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2048xi32, #blocked>
    %2 = arith.cmpi slt, %1, %cst_0 : tensor<1x2048xi32, #blocked>
    %3 = arith.xori %2, %cst : tensor<1x2048xi1, #blocked>
    %4 = tt.broadcast %3 : tensor<1x2048xi1, #blocked> -> tensor<2x2048xi1, #blocked>
    // CHECK: tt.assert {{.*}} : tensor<2x2048xi1, #[[$BLOCKED]]>
    tt.assert %4, "index out of bounds: 0 <= tmp28 < 32" : tensor<2x2048xi1, #blocked>
    %5 = tt.broadcast %1 : tensor<1x2048xi32, #blocked> -> tensor<2x2048xi32, #blocked>
    %6 = arith.extsi %5 : tensor<2x2048xi32, #blocked> to tensor<2x2048xi64, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<2x2048x!tt.ptr<f16>, #blocked>
    %8 = tt.addptr %7, %6 : tensor<2x2048x!tt.ptr<f16>, #blocked>, tensor<2x2048xi64, #blocked>
    // COM: the following conversion should be removed because the layout used by the load operation is backward propagated.
    %9 = ttg.convert_layout %8 : tensor<2x2048x!tt.ptr<f16>, #blocked> -> tensor<2x2048x!tt.ptr<f16>, #blocked1>
    %10 = tt.load %9 evictionPolicy = evict_last : tensor<2x2048x!tt.ptr<f16>, #blocked1>
    tt.print " " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<2x2048xf16, #blocked1>
    tt.return
  }
}
