// RUN: triton-opt %s -test-print-stride-info -split-input-file -o %t 2>&1 | FileCheck %s

// The stride deduction for a signed remainder needs a dividend the analysis can
// prove non-negative.
// `%base` is a function argument, so nothing bounds it.

// CHECK-LABEL: @rem_unknown_sign
tt.func @rem_unknown_sign(%base: i32) {
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<512> : tensor<128xi32>
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// Assuming `%base >= 0` makes the dividend provably non-negative, so the
// stride survives the remainder: the window is 128-aligned and 128 wide, so
// it cannot reach a multiple of 512.

// CHECK-LABEL: @rem_assumed_non_negative
tt.func @rem_assumed_non_negative(%base: i32) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<512> : tensor<128xi32>
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}
