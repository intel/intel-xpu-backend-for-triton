// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that masks with sge (>=) predicates are correctly classified.
  // COM: The mask (IV + range(0,32)) >= 0 is always true since IV starts at 0
  // COM: and make_range produces non-negative values.
  tt.func public @test_sge_mask_always_true(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst_zero = arith.constant dense<0> : tensor<1x32xi32>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range_2d = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>

    %result = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32
        iter_args(%acc = %cst) -> (tensor<32x32xf16>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<1x32xi32>
      %offset = arith.addi %iv_splat, %range_2d : tensor<1x32xi32>
      // sge mask: (IV + range(0,32)) >= 0 — always true.
      %mask_1d = arith.cmpi sge, %offset, %cst_zero : tensor<1x32xi32>
      %mask = tt.broadcast %mask_1d : tensor<1x32xi1> -> tensor<32x32xi1>

      %loaded = tt.load %base, %mask, %cst : tensor<32x32x!tt.ptr<f16>>
      scf.yield %loaded : tensor<32x32xf16>
    }
    tt.return
  }

  // CHECK: tt.func public @test_sge_mask_always_true
  // COM: The sge mask is always true (min element = 0 + 0 = 0 >= 0).
  // CHECK: scf.for
  // CHECK:   tt.load {{%[0-9]+}} : tensor<32x32x!tt.ptr<f16>>
  // CHECK: }
}
