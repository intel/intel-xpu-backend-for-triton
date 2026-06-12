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

  // -----

  // COM: Regression test for issue #6871.
  // COM: Ensure arith.select with AlwaysFalse condition and AlwaysTrue true-value folds to false-value, not true-value.
  // CHECK-LABEL: issue_6871
  tt.func public @issue_6871(%arg0: tensor<32x!tt.ptr<f32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant dense<0> : tensor<32xi32>
    %c64 = arith.constant dense<64> : tensor<32xi32>
    %cst_false = arith.constant dense<false> : tensor<32xi1>
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    // CHECK: [[CST_FALSE:%.+]] = arith.constant dense<false> : tensor<32xi1>

    %result = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%carry = %cst_false) -> (tensor<32xi1>) : i32 {
      // Offsets = IV (scalar splatted) + make_range [0, 32)
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>

      // mask_true: offsets < 64 → AlwaysTrue for IV in [0, 32)
      %mask_true = arith.cmpi slt, %offsets, %c64 : tensor<32xi32>

      // mask_false: offsets < 0 → AlwaysFalse for IV in [0, 32)
      %mask_false = arith.cmpi slt, %offsets, %c0 : tensor<32xi32>

      // Load with mask_true (always true, should be optimized to unmasked)
      %loaded = tt.load %arg0, %mask_true, %cst_zero : tensor<32x!tt.ptr<f32>>

      // Select with mask_false as condition, mask_true as true-value, cst_false as false-value
      // Since mask_false is AlwaysFalse, this MUST fold to cst_false, NOT mask_true
      %selected = arith.select %mask_false, %mask_true, %cst_false : tensor<32xi1>, tensor<32xi1>

      scf.yield %selected : tensor<32xi1>
    }
    // COM: mask_true is AlwaysTrue, so the load is rewritten without a mask.
    // CHECK: tt.load %arg0 :
    // CHECK: scf.yield [[CST_FALSE]] : tensor<32xi1>

    tt.return
  }
}
