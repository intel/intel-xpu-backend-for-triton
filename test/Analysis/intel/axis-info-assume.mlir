// RUN: triton-opt %s -test-print-axis-info -split-input-file -o %t 2>&1 | FileCheck %s

// The contiguity deduction for a signed remainder needs a dividend the analysis can
// prove non-negative. `%base` is a function argument, so nothing bounds it and the
// deduction is declined.

// CHECK-LABEL: @rem_unknown_sign
tt.func @rem_unknown_sign(%base: i32, %M: i32 {tt.divisibility = 16 : i32}) {
  %c128 = arith.constant 128 : i32
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %M : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %offs, %mod : tensor<128xi32>
  tt.return
}

// -----

// Assuming `%base >= 0` makes the whole dividend provably non-negative -
// multiplication and addition both preserve it - so the deduction is taken:
// gcd(contiguity 128, divisibility 128, modulus divisibility 16) = 16.

// CHECK-LABEL: @rem_assumed_non_negative
tt.func @rem_assumed_non_negative(%base: i32, %M: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %M : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %offs, %mod : tensor<128xi32>
  tt.return
}
