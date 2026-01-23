// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// CHECK-NOT: ttg.convert_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 8], order = [1, 0]}>
module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @remove_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<15> : tensor<2x2048xi64, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<2x2048xi64, #blocked>
    %cst_1 = arith.constant dense<true> : tensor<1x2048xi1, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<2x2048xi64, #blocked>
    %cst_3 = arith.constant dense<2048> : tensor<1x2048xi32, #blocked>
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2048xi32, #blocked>
    %2 = arith.cmpi slt, %1, %cst_3 : tensor<1x2048xi32, #blocked>
    %3 = tt.broadcast %1 : tensor<1x2048xi32, #blocked> -> tensor<2x2048xi32, #blocked>
    %4 = arith.extsi %3 : tensor<2x2048xi32, #blocked> to tensor<2x2048xi64, #blocked>
    %5 = arith.cmpi slt, %4, %cst : tensor<2x2048xi64, #blocked>
    %6 = arith.select %5, %4, %cst : tensor<2x2048xi1, #blocked>, tensor<2x2048xi64, #blocked>
    %7 = arith.cmpi sge, %6, %cst_2 : tensor<2x2048xi64, #blocked>
    %8 = arith.cmpi slt, %6, %cst_0 : tensor<2x2048xi64, #blocked>
    %9 = arith.andi %7, %8 : tensor<2x2048xi1, #blocked>
    %10 = arith.xori %2, %cst_1 : tensor<1x2048xi1, #blocked>
    %11 = tt.broadcast %10 : tensor<1x2048xi1, #blocked> -> tensor<2x2048xi1, #blocked>
    %12 = arith.ori %9, %11 : tensor<2x2048xi1, #blocked>
    tt.assert %7, "index out of bounds: 0 <= tmp28 < 32" : tensor<2x2048xi1, #blocked>
    %13 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<2x2048x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %6 : tensor<2x2048x!tt.ptr<f16>, #blocked>, tensor<2x2048xi64, #blocked>

    // COM: the following conversion layout should be backward to tt.assert.
    %15 = ttg.convert_layout %14 : tensor<2x2048x!tt.ptr<f16>, #blocked> -> tensor<2x2048x!tt.ptr<f16>, #blocked1>
    %16 = tt.load %15 evictionPolicy = evict_last : tensor<2x2048x!tt.ptr<f16>, #blocked1>
    tt.print " " {hex = false, isSigned = array<i32: 0>} : %16 : tensor<2x2048xf16, #blocked1>
    tt.return
  }
}
