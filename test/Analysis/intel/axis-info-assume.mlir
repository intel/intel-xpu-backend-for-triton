// RUN: triton-opt %s -test-print-axis-info -split-input-file -o %t 2>&1 | FileCheck %s

// COM: The contiguity deduction for a signed remainder needs a dividend the analysis can
// COM: prove non-negative. `%base` is a function argument, so nothing bounds it and the
// COM: deduction is declined.

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

// COM: Assuming `%base >= 0` makes the whole dividend provably non-negative -
// COM: multiplication and addition both preserve it - so the deduction is taken:
// COM: gcd(contiguity 128, divisibility 128, modulus divisibility 16) = 16.

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

// COM: The same for the constancy deduction of a signed division: nothing bounds
// COM: `%base`, so it is declined.

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

// COM: With `%base >= 0` assumed the division keeps its constancy:
// COM: gcd(contiguity 128, divisibility 128, divisor divisibility 16) = 16.

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

// COM: An assume only counts where it provably ran first; this one sits after the
// COM: division, so the dividend's sign is unknown at the point that matters.

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

// COM: An `scf.for` counter runs over `lb + k * step`, so non-negative bounds make
// COM: every value it takes non-negative.

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

// COM: A counter that can start below zero is not non-negative. `%lb` carries a
// COM: divisibility so that only its sign is unknown - without one the deduction
// COM: would collapse on divisibility and the sign check would never be consulted.

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

// COM: An unsigned remainder is not non-negative on its own: with `%a` = 0xffffffc0
// COM: and `%d` = 0xfffffff0 the result is -64 to the signed remainder below.

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

// COM: Either operand bounds it. A non-negative divisor keeps the remainder below
// COM: itself; a non-negative dividend keeps it at or below that instead.

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

// COM: An unsigned division by 1 is the identity, so it carries a negative dividend
// COM: through; by 2 or more the quotient's top bit is clear.

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

// -----

// COM: The same counter after `convert-scf-to-cf`, as a header block argument: it
// COM: reaches the header with 0 from outside and with the incremented counter from
// COM: the back edge, so it stays non-negative.

// CHECK-LABEL: @rem_branch_counter
tt.func @rem_branch_counter(%rnumel: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  cf.br ^header(%c0 : i32)
^header(%roffset: i32):
  %cond = arith.cmpi slt, %roffset, %rnumel : i32
  cf.cond_br %cond, ^body, ^exit
^body:
  %splat = tt.splat %roffset : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %rindex = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %HW : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %rindex, %mod : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [16]
  %div = arith.divsi %rindex, %mod : tensor<128xi32>
  %next = arith.addi %roffset, %c128 : i32
  cf.br ^header(%next : i32)
^exit:
  tt.return
}

// -----

// COM: Entered with a lower bound of unknown sign: the edge that is the base case of
// COM: the induction cannot be proved, so the deduction is declined.

// CHECK-LABEL: @rem_branch_counter_negative_start
tt.func @rem_branch_counter_negative_start(%lb: i32 {tt.divisibility = 128 : i32}, %rnumel: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c128 = arith.constant 128 : i32
  cf.br ^header(%lb : i32)
^header(%roffset: i32):
  %cond = arith.cmpi slt, %roffset, %rnumel : i32
  cf.cond_br %cond, ^body, ^exit
^body:
  %splat = tt.splat %roffset : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %rindex = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %HW : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %rindex, %mod : tensor<128xi32>
  %next = arith.addi %roffset, %c128 : i32
  cf.br ^header(%next : i32)
^exit:
  tt.return
}

// -----

// COM: The back edge is checked too: this counter is entered with a bound that
// COM: is assumed non-negative, so the entry edge holds, but it counts down and
// COM: `arith.subi` says nothing about its result's sign.

// CHECK-LABEL: @rem_branch_counter_counting_down
tt.func @rem_branch_counter_counting_down(%rnumel: i32 {tt.divisibility = 128 : i32}, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %nonneg = arith.cmpi sge, %rnumel, %c0 : i32
  llvm.intr.assume %nonneg : i1
  cf.br ^header(%rnumel : i32)
^header(%roffset: i32):
  %cond = arith.cmpi sgt, %roffset, %c0 : i32
  cf.cond_br %cond, ^body, ^exit
^body:
  %splat = tt.splat %roffset : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %rindex = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %HW : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %rindex, %mod : tensor<128xi32>
  %next = arith.subi %roffset, %c128 : i32
  cf.br ^header(%next : i32)
^exit:
  tt.return
}

// -----

// COM: An assume in the entry block covers a use in any other block of the
// COM: region. This is the form `convert-scf-to-cf` leaves.

// CHECK-LABEL: @rem_assumed_in_entry_block
tt.func @rem_assumed_in_entry_block(%base: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %nonneg = arith.cmpi sge, %base, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %scaled = arith.muli %base, %c128 : i32
  cf.br ^body
^body:
  %splat = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %mod = tt.splat %HW : i32 -> tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %offs, %mod : tensor<128xi32>
  tt.return
}

// -----

// COM: Both operands of the `arith.maxsi` walk to `%counter`, whose entry edge
// COM: cannot be proved. A shared hypothesis set would let the second operand
// COM: find what the failed first left behind and report contiguity 16 here.

// CHECK-LABEL: @rem_max_of_unproven_counter
tt.func @rem_max_of_unproven_counter(%lb: i32 {tt.divisibility = 128 : i32}, %rnumel: i32) {
  %c128 = arith.constant 128 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  cf.br ^header(%lb : i32)
^header(%counter: i32):
  %cond = arith.cmpi slt, %counter, %rnumel : i32
  cf.cond_br %cond, ^body, ^exit
^body:
  %max = arith.maxsi %counter, %counter : i32
  %splat = tt.splat %max : i32 -> tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  %next = arith.addi %counter, %c128 : i32
  cf.br ^header(%next : i32)
^exit:
  tt.return
}

// -----

// COM: `arith.minsi` can deliver either operand, so it needs both to be
// COM: non-negative, while `arith.maxsi` is at or above either and so needs only
// COM: one. `%p` is bounded by an assume in the `x > -1` form, `%n` by nothing.
// COM: The declined case comes first so that neither check can slide onto the
// COM: other's line.

// CHECK-LABEL: @rem_over_min_max
tt.func @rem_over_min_max(%n: i32 {tt.divisibility = 128 : i32}, %p: i32 {tt.divisibility = 128 : i32}) {
  %cm1 = arith.constant -1 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nonneg = arith.cmpi sgt, %p, %cm1 : i32
  llvm.intr.assume %nonneg : i1
  %min = arith.minsi %n, %p : i32
  %minSplat = tt.splat %min : i32 -> tensor<128xi32>
  %minOffs = arith.addi %minSplat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %minRem = arith.remsi %minOffs, %cst : tensor<128xi32>
  %max = arith.maxsi %n, %p : i32
  %maxSplat = tt.splat %max : i32 -> tensor<128xi32>
  %maxOffs = arith.addi %maxSplat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %maxRem = arith.remsi %maxOffs, %cst : tensor<128xi32>
  tt.return
}

// -----

// COM: Either arm of an `arith.select` can be the one that runs, so both have to
// COM: be non-negative.

// CHECK-LABEL: @rem_over_select
tt.func @rem_over_select(%n: i32 {tt.divisibility = 128 : i32}, %p: i32 {tt.divisibility = 128 : i32}, %cond: i1) {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nonneg = arith.cmpi sge, %p, %c0 : i32
  llvm.intr.assume %nonneg : i1
  %sel = arith.select %cond, %p, %n : i32
  %selSplat = tt.splat %sel : i32 -> tensor<128xi32>
  %selOffs = arith.addi %selSplat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %selRem = arith.remsi %selOffs, %cst : tensor<128xi32>
  tt.return
}

// -----

// COM: A sign extension preserves the value, so it carries the sign of its
// COM: input - including an unknown one. Only `arith.extui` clears the top bit
// COM: whatever its input was.

// CHECK-LABEL: @rem_over_sign_extension
tt.func @rem_over_sign_extension(%n: i32 {tt.divisibility = 128 : i32}) {
  %cst = arith.constant dense<16> : tensor<128xi64>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %splat = tt.splat %n : i32 -> tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  %ext = arith.extsi %offs : tensor<128xi32> to tensor<128xi64>
  // CHECK: arith.remsi {{.*}} => contiguity = [1], divisibility = [1], constancy = [1]
  %rem = arith.remsi %ext, %cst : tensor<128xi64>
  tt.return
}

// -----

// COM: An arithmetic shift right preserves the sign bit, so a non-negative
// COM: input keeps the deduction. `%q` is bounded by an assume in the swapped
// COM: `0 <= x` form.

// CHECK-LABEL: @rem_over_shift_right
tt.func @rem_over_shift_right(%q: i32 {tt.divisibility = 128 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nonneg = arith.cmpi sle, %c0, %q : i32
  llvm.intr.assume %nonneg : i1
  %shift = arith.shrsi %q, %c1 : i32
  %splat = tt.splat %shift : i32 -> tensor<128xi32>
  %offs = arith.addi %splat, %range : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => contiguity = [16], divisibility = [16], constancy = [1]
  %rem = arith.remsi %offs, %cst : tensor<128xi32>
  tt.return
}
