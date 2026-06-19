// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that a K-dimension mask using an iter_arg (not the canonical IV)
  // COM: is correctly classified as always-true via getIVEquivalentRange.
  // COM: The iter_arg off_k starts at 0 and increments by 32 (same as the loop
  // COM: IV), so its range is [0, 96]. Max element = 96 + 31 = 127 < 4096.
  tt.func public @test_iter_arg_k_mask(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>

    %k_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %k_range_i64 = arith.extsi %k_range : tensor<32xi32> to tensor<32xi64>
    %base_splat = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>

    // Loop with off_k as iter_arg (same as IV: init=0, step=32).
    %result = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%off_k = %c0_i32) -> (i32) : i32 {
      %off_k_i64 = arith.extsi %off_k : i32 to i64
      %k_splat = tt.splat %off_k_i64 : i64 -> tensor<32xi64>
      %k_offset = arith.addi %k_splat, %k_range_i64 : tensor<32xi64>

      // K boundary check: (off_k + range(0,32)) < splat(4096).
      // Always true since off_k in [0, 96], max element = 127 < 4096.
      %k_ub = tt.splat %c4096_i64 : i64 -> tensor<32xi64>
      %k_mask = arith.cmpi slt, %k_offset, %k_ub : tensor<32xi64>

      %loaded = tt.load %base_splat, %k_mask, %cst : tensor<32x!tt.ptr<f16>>

      %next_off_k = arith.addi %off_k, %c32_i32 : i32
      scf.yield %next_off_k : i32
    }
    tt.return
  }

  // CHECK: tt.func public @test_iter_arg_k_mask
  // COM: The mask is proven always-true and directly removed.
  // CHECK: scf.for
  // CHECK:   tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
  // CHECK: }
}
