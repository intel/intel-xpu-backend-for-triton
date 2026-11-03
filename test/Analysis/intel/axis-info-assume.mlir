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

// -----

// The same for the constancy deduction of a signed division: nothing bounds
// `%base`, so it is declined.

// CHECK-LABEL: @div_unknown_sign
tt.func @div_unknown_sign(%base: i32, %M: i32 {tt.divisibility = 16 : i32}) {
  %c128 = arith.constant 128 : i32
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %M : i32 -> tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %div = arith.divsi %offs, %mod : tensor<128xi32>
  tt.return
}

// -----

// With `%base >= 0` assumed the division keeps its constancy:
// gcd(contiguity 128, divisibility 128, divisor divisibility 16) = 16.

// CHECK-LABEL: @div_assumed_non_negative
tt.func @div_assumed_non_negative(%base: i32, %M: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %M : i32 -> tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [16]
  %div = arith.divsi %offs, %mod : tensor<128xi32>
  tt.return
}

// -----

// An assume only counts where it provably ran first; this one sits after the
// division, so the dividend's sign is unknown at the point that matters.

// CHECK-LABEL: @div_assumed_too_late
tt.func @div_assumed_too_late(%base: i32, %M: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %scaled = arith.muli %base, %c128 : i32
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %M : i32 -> tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %div = arith.divsi %offs, %mod : tensor<128xi32>
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  tt.return
}

// -----

// An `scf.for` counter runs over `lb + k * step`, so non-negative bounds make
// every value it takes non-negative.

// CHECK-LABEL: @rem_loop_counter
tt.func @rem_loop_counter(%rnumel: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  scf.for %roffset = %c0 to %rnumel step %c128  : i32 {
    %splat = tt.splat %roffset : i32 -> tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %rindex = arith.addi %splat, %range : tensor<128xi32>
    %mod = tt.splat %HW : i32 -> tensor<128xi32>
    // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
    %rem = arith.remsi %rindex, %mod : tensor<128xi32>
    // CHECK: arith.divsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [16]
    %div = arith.divsi %rindex, %mod : tensor<128xi32>
  }
  tt.return
}

// -----

// A counter that can start below zero is not non-negative. `%lb` carries a
// divisibility so that only its sign is unknown - without one the deduction
// would collapse on divisibility and the sign check would never be consulted.

// CHECK-LABEL: @rem_loop_counter_negative_start
tt.func @rem_loop_counter_negative_start(%lb: i32 {tt.divisibility = 128 : i32}, %rnumel: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c128 = arith.constant 128 : i32
  scf.for %roffset = %lb to %rnumel step %c128  : i32 {
    %splat = tt.splat %roffset : i32 -> tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %rindex = arith.addi %splat, %range : tensor<128xi32>
    %mod = tt.splat %HW : i32 -> tensor<128xi32>
    // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
    %rem = arith.remsi %rindex, %mod : tensor<128xi32>
  }
  tt.return
}

// -----

// An unsigned remainder is not non-negative on its own: with `%a` = 0xffffffc0
// and `%d` = 0xfffffff0 the result is -64 to the signed remainder below.

// CHECK-LABEL: @rem_over_unsigned_remainder
tt.func @rem_over_unsigned_remainder(%a: i32 {tt.divisibility = 16 : i32}, %d: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<16> : tensor<128xi32>
  %u = arith.remui %a, %d : i32
  %splat = tt.splat %u : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// Either operand bounds it. A non-negative divisor keeps the remainder below
// itself; a non-negative dividend keeps it at or below that instead.

// CHECK-LABEL: @rem_over_unsigned_remainder_bounded_divisor
tt.func @rem_over_unsigned_remainder_bounded_divisor(%a: i32 {tt.divisibility = 16 : i32}, %d: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %nonneg = arith.cmpi sge, %d, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %u = arith.remui %a, %d : i32
  %splat = tt.splat %u : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @rem_over_unsigned_remainder_bounded_dividend
tt.func @rem_over_unsigned_remainder_bounded_dividend(%a: i32 {tt.divisibility = 16 : i32}, %d: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %nonneg = arith.cmpi sge, %a, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %u = arith.remui %a, %d : i32
  %splat = tt.splat %u : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}

// -----

// An unsigned division by 1 is the identity, so it carries a negative dividend
// through; by 2 or more the quotient's top bit is clear.

// CHECK-LABEL: @rem_over_unsigned_division
tt.func @rem_over_unsigned_division(%a: i32 {tt.divisibility = 16 : i32}) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %by1 = arith.divui %a, %c1 : i32
  %splat1 = tt.splat %by1 : i32 -> tensor<128xi32>
  %offs1 = arith.addi %splat1, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem1 = arith.remsi %offs1, %cst : tensor<128xi32>
  %by2 = arith.divui %a, %c2 : i32
  %splat2 = tt.splat %by2 : i32 -> tensor<128xi32>
  %offs2 = arith.addi %splat2, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [8], divisibility = [8], constancy = [1]
  %rem2 = arith.remsi %offs2, %cst : tensor<128xi32>
  tt.return
}
