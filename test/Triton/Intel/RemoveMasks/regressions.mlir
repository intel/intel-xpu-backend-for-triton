// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Regression test for issue #4556.
  // COM: Ensure test compiles without triggering an assertion.
  tt.func public @issue_4556(%arg0: tensor<16x1x1x!tt.ptr<f32>>) {
    // CHECK-LABEL: issue_4556
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c196_i32 = arith.constant 196 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x1x1xf32>
    %cst_196 = arith.constant dense<196> : tensor<1x1x1xi32>
    scf.for %iv = %c0_i32 to %c196_i32 step %c1_i32 : i32 {
      %0 = tt.splat %iv : i32 -> tensor<1x1x1xi32>
      %1 = arith.cmpi slt, %0, %cst_196 : tensor<1x1x1xi32>
      %mask = tt.broadcast %1 : tensor<1x1x1xi1> -> tensor<16x1x1xi1>
      %2 = tt.load %arg0, %mask, %cst_0 : tensor<16x1x1x!tt.ptr<f32>>
      scf.yield
    }
    tt.return
  }
}
