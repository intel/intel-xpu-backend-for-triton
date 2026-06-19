// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that RemoveMasks can trace through arith.extsi when classifying
  // COM: loop-dependent masks. The mask pattern uses extsi(IV) before splat,
  // COM: which previously blocked getFinalValue from resolving to the IV.
  // COM: Pattern: cmpi slt, (splat(extsi(IV)) + extsi(make_range)), splat(K)
  tt.func public @test_extsi_iv_mask(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c576_i32 = arith.constant 576 : i32
    %c576_i64 = arith.constant 576 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16>

    %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
    %range_2d = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
    %ub_splat = tt.splat %c576_i64 : i64 -> tensor<1x32xi64>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>

    %result = scf.for %iv = %c0_i32 to %c576_i32 step %c32_i32
        iter_args(%acc = %cst) -> (tensor<32x32xf16>) : i32 {
      // The IV is i32, extended to i64 before splat — this is the extsi
      // that getFinalValue must peel through.
      %iv_i64 = arith.extsi %iv : i32 to i64
      %iv_splat = tt.splat %iv_i64 : i64 -> tensor<32xi64>
      %offset_1d = arith.addi %iv_splat, %range : tensor<32xi64>
      %offset = tt.expand_dims %offset_1d {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>

      // Mask: (splat(extsi(IV)) + extsi(range)) < splat(576)
      %mask_1d = arith.cmpi slt, %offset, %ub_splat : tensor<1x32xi64>
      %mask = tt.broadcast %mask_1d : tensor<1x32xi1> -> tensor<32x32xi1>

      %loaded = tt.load %base, %mask, %cst : tensor<32x32x!tt.ptr<f16>>
      scf.yield %loaded : tensor<32x32xf16>
    }
    tt.return
  }

  // CHECK: tt.func public @test_extsi_iv_mask
  // COM: The mask (extsi(IV) + extsi(range(0,32))) < 576 is always true since
  // COM: IV ranges [0, 544] and max element = 544 + 31 = 575 < 576.
  // COM: Without ExtSI peeling in getFinalValue, the IV cannot be resolved
  // COM: and the mask would remain.
  // CHECK: scf.for
  // CHECK:   tt.load {{%[0-9]+}} : tensor<32x32x!tt.ptr<f16>>
  // CHECK: }
}
