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

// -----

// The same for the stride deduction of a signed division. Nothing bounds
// `%base`, so it is declined.

// CHECK-LABEL: @div_unknown_sign
tt.func @div_unknown_sign(%base: i32) {
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<4> : tensor<128xi32>
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %strided = arith.muli %range, %cst : tensor<128xi32>
  %offs = arith.addi %splat, %strided : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [-1]
  %div = arith.divsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// With `%base >= 0` assumed the stride survives the division: a stride of 4
// divided by 4 is a stride of 1.

// CHECK-LABEL: @div_assumed_non_negative
tt.func @div_assumed_non_negative(%base: i32) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<4> : tensor<128xi32>
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %strided = arith.muli %range, %cst : tensor<128xi32>
  %offs = arith.addi %splat, %strided : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [1]
  %div = arith.divsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// An `scf.for` counter over non-negative bounds is non-negative.

// CHECK-LABEL: @loop_counter
tt.func @loop_counter(%rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<512> : tensor<128xi32>
  scf.for %roffset = %c0 to %rnumel step %c128  : i32 {
    %splat = tt.splat %roffset : i32 -> tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %rindex = arith.addi %splat, %range : tensor<128xi32>
    // CHECK: arith.remsi {{.*}} => stride = [1]
    %rem = arith.remsi %rindex, %cst : tensor<128xi32>
  }
  tt.return
}

// -----

// A counter that can start below zero is not non-negative. `%lb` carries a
// divisibility so that only its sign is unknown.

// CHECK-LABEL: @loop_counter_negative_start
tt.func @loop_counter_negative_start(%lb: i32 {tt.divisibility = 128 : i32}, %rnumel: i32) {
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<512> : tensor<128xi32>
  scf.for %roffset = %lb to %rnumel step %c128  : i32 {
    %splat = tt.splat %roffset : i32 -> tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %rindex = arith.addi %splat, %range : tensor<128xi32>
    // CHECK: arith.remsi {{.*}} => stride = [-1]
    %rem = arith.remsi %rindex, %cst : tensor<128xi32>
  }
  tt.return
}
