// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Regression test for issue #4556.
  // COM: Ensure test compiles without triggering an assertion.
  tt.func public @issue_4556(%arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: issue_4556
    %c1_i32 = arith.constant 1 : i32
    %c196_i32 = arith.constant 196 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x1x1xf32>
    %cst_1 = arith.constant dense<196> : tensor<16x1x1xi32>
    %cst_6 = arith.constant dense<196> : tensor<1x1x1xi32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<16x16x1xf32>
    %cst_8 = arith.constant dense<16> : tensor<16x1x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id y : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<16x1xi32> -> tensor<16x1x1xi32>
    %5 = tt.splat %1 : i32 -> tensor<16x1x1xi32>
    %6 = arith.addi %5, %4 : tensor<16x1x1xi32>
    %16 = arith.divsi %6, %cst_8 : tensor<16x1x1xi32>
    %17:2 = scf.for %arg9 = %c0_i32 to %c196_i32 step %c1_i32 iter_args(%arg10 = %cst_7, %arg11 = %cst_7) -> (tensor<16x16x1xf32>, tensor<16x16x1xf32>)  : i32 {
      %36 = tt.splat %arg9 : i32 -> tensor<1x1x1xi32>
      %37 = arith.cmpi slt, %36, %cst_6 : tensor<1x1x1xi32>
      %72 = arith.muli %16, %cst_1 : tensor<16x1x1xi32>
      %73 = tt.splat %arg9 : i32 -> tensor<16x1x1xi32>
      %74 = arith.addi %73, %72 : tensor<16x1x1xi32>
      %75 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x1x1x!tt.ptr<f32>>
      %76 = tt.addptr %75, %74 : tensor<16x1x1x!tt.ptr<f32>>, tensor<16x1x1xi32>
      %77 = tt.broadcast %37 : tensor<1x1x1xi1> -> tensor<16x1x1xi1>
      %78 = tt.load %76, %77, %cst_0 evictionPolicy = evict_last : tensor<16x1x1x!tt.ptr<f32>>
      scf.yield %arg10, %arg11 : tensor<16x16x1xf32>, tensor<16x16x1xf32>
    }
    tt.return
  }
}
