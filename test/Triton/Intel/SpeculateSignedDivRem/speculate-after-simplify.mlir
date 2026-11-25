// RUN: triton-opt %s -split-input-file -triton-intel-simplify-signed-arithmetic -triton-intel-speculate-signed-div-rem | FileCheck %s

// COM: The pipeline order matters: triton-intel-simplify-signed-arithmetic proves
// COM: what it can and converts without a check, so this pass only sees the
// COM: operations that need one. A provably non-negative dividend must therefore
// COM: end up unsigned with no assertion.
module {
tt.func public @provable_dividend_needs_no_assert() -> (tensor<128xi32>, tensor<128xi32>) {
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %rem = arith.remsi %range, %cst : tensor<128xi32>
  %div = arith.divsi %range, %cst : tensor<128xi32>
  tt.return %rem, %div : tensor<128xi32>, tensor<128xi32>
}
// CHECK-LABEL: @provable_dividend_needs_no_assert
// CHECK-NOT:     tt.assert
// CHECK:         arith.remui
// CHECK:         arith.divui
// CHECK-NOT:     tt.assert
}

// -----

// COM: An unprovable dividend of the same shape does get an assertion, so the
// COM: distinction above is drawn by the prover, not by the match conditions.
// COM: The sign here depends on a kernel argument, which is the case the
// COM: speculation exists for: neither provably non-negative nor provably
// COM: negative.
module {
tt.func public @unprovable_dividend_gets_an_assert(%arg0: i32) -> tensor<128xi32> {
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %splat = tt.splat %arg0 : i32 -> tensor<128xi32>
  %off = arith.muli %splat, %cst : tensor<128xi32>
  %base = arith.addi %range, %off : tensor<128xi32>
  %div = arith.divsi %base, %cst : tensor<128xi32>
  tt.return %div : tensor<128xi32>
}
// CHECK-LABEL: @unprovable_dividend_gets_an_assert
// CHECK:         tt.assert
// CHECK:         arith.divui
}
