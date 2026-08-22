// RUN: triton-opt %s -split-input-file -triton-intel-speculate-signed-div-rem | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive tests - the AxisInfo deduction applies, so speculate.
//===----------------------------------------------------------------------===//

// COM: One contiguous dividend feeding both a division and a
// COM: remainder by the same constant. Both are converted, and the assertion is
// COM: emitted only once for the shared dividend.
// COM:
// COM: The dividend is non-negative as soon as the scalar offset is, so that is
// COM: the fact the prover is left with and the fact that gets checked - one
// COM: scalar comparison instead of a comparison over the whole index tensor.
module {
tt.func public @div_and_rem_share_one_assert(%arg0: i32) -> (tensor<1x64xi32>, tensor<1x64xi32>) {
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %c64_i32 = arith.constant 64 : i32
  %scaled = arith.muli %arg0, %c64_i32 : i32
  %off = tt.splat %scaled : i32 -> tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %exp = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %idx = arith.addi %exp, %off : tensor<1x64xi32>
  %rem = arith.remsi %idx, %cst : tensor<1x64xi32>
  %div = arith.divsi %idx, %cst : tensor<1x64xi32>
  tt.return %rem, %div : tensor<1x64xi32>, tensor<1x64xi32>
}
// CHECK-LABEL: @div_and_rem_share_one_assert
// CHECK-SAME:    (%[[ARG0:.*]]: i32)
// CHECK:         %[[CST:.*]] = arith.constant dense<128> : tensor<1x64xi32>
// CHECK:         %[[IDX:.*]] = arith.addi
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         %[[COND:.*]] = arith.cmpi sge, %[[ARG0]], %[[ZERO]] : i32
// CHECK:         tt.assert %[[COND]]
// CHECK-NOT:     tt.assert
// CHECK:         %[[REM:.*]] = arith.remui %[[IDX]], %[[CST]] : tensor<1x64xi32>
// CHECK:         %[[DIV:.*]] = arith.divui %[[IDX]], %[[CST]] : tensor<1x64xi32>
// CHECK:         tt.return %[[REM]], %[[DIV]]
}

// -----

// COM: The dividend is a bare range, so it is provably non-negative and there is
// COM: nothing left for a runtime check to establish. It converts outright, with
// COM: no assertion - checking a tautology would cost the whole benefit.
module {
tt.func public @provably_non_negative_dividend() -> tensor<1x64xi32> {
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %idx = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %rem = arith.remsi %idx, %cst : tensor<1x64xi32>
  tt.return %rem : tensor<1x64xi32>
}
// CHECK-LABEL: @provably_non_negative_dividend
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     tt.assert
// CHECK:         arith.remui
}

// -----

// COM: Same provable dividend, but used inside an scf.if nested in an scf.for -
// COM: the one place an assertion cannot be hoisted to, so a speculated operation
// COM: here would be left signed (see @dividend_in_conditional). This one still
// COM: converts, because a proven dividend emits no assertion for a loop to
// COM: contain. Pins that the hoistability bail-out gates the checks, not the
// COM: conversion.
module {
tt.func public @provable_dividend_in_loop(%c: i1, %ub: i32) -> tensor<1x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %init = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %init) -> (tensor<1x64xi32>) : i32 {
    %sum = scf.if %c -> (tensor<1x64xi32>) {
      %rem = arith.remsi %init, %cst : tensor<1x64xi32>
      %s = arith.addi %acc, %rem : tensor<1x64xi32>
      scf.yield %s : tensor<1x64xi32>
    } else {
      scf.yield %acc : tensor<1x64xi32>
    }
    scf.yield %sum : tensor<1x64xi32>
  }
  tt.return %res : tensor<1x64xi32>
}
// CHECK-LABEL: @provable_dividend_in_loop
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     tt.assert
// CHECK:         arith.remui
}

// -----

// COM: The dividend is a loop-carried index advanced by
// COM: `(load(next) - load(cur)) * 128 - 64`, so it stays non-negative only if
// COM: the loaded block indices are in increasing order. That is a property of
// COM: the data in the buffer, not of the IR, so no prover can show it.
// COM:
// COM: The check therefore has to happen at runtime, but a tt.assert in the body
// COM: would stop IGC from optimizing the loop. So the check is folded into a
// COM: loop-carried flag - one arith.andi per iteration - and asserted once
// COM: after the loop.
module {
tt.func public @loop_carried_dividend(%ub: i32, %ptr: !tt.ptr<i32>) -> tensor<1x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %init = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %res:2 = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%idx = %init, %acc = %init)
      -> (tensor<1x64xi32>, tensor<1x64xi32>) : i32 {
    %rem = arith.remsi %idx, %cst : tensor<1x64xi32>
    %sum = arith.addi %acc, %rem : tensor<1x64xi32>
    %l = tt.load %ptr : !tt.ptr<i32>
    %scaled = arith.muli %l, %c128_i32 : i32
    %advance = arith.subi %scaled, %c64_i32 : i32
    %splat = tt.splat %advance : i32 -> tensor<1x64xi32>
    %next = arith.addi %idx, %splat : tensor<1x64xi32>
    scf.yield %next, %sum : tensor<1x64xi32>, tensor<1x64xi32>
  }
  tt.return %res#1 : tensor<1x64xi32>
}
// CHECK-LABEL: @loop_carried_dividend
// CHECK:         %[[TRUE:.*]] = arith.constant dense<true> : tensor<1x64xi1>
// CHECK:         %[[RES:.*]]:3 = scf.for {{.*}} iter_args(%[[IDX:.*]] = %{{.*}}, %{{.*}} = %{{.*}}, %[[FLAG:.*]] = %[[TRUE]])
// CHECK-SAME:        -> (tensor<1x64xi32>, tensor<1x64xi32>, tensor<1x64xi1>)
// CHECK:           %[[COND:.*]] = arith.cmpi sge, %[[IDX]], %{{.*}} : tensor<1x64xi32>
// CHECK:           arith.remui %[[IDX]], %{{.*}} : tensor<1x64xi32>
// CHECK-NOT:       tt.assert
// CHECK:           %[[AND:.*]] = arith.andi %[[FLAG]], %[[COND]] : tensor<1x64xi1>
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %[[AND]]
// CHECK:         }
// CHECK:         tt.assert %[[RES]]#2
}

// -----

// COM: The flag is accumulated once per loop level, so the assertion ends up
// COM: after the outermost loop and no loop body contains a tt.assert.
module {
tt.func public @nested_loops(%ub: i32, %ptr: !tt.ptr<i32>) -> tensor<1x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %init = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %outer = scf.for %j = %c0_i32 to %ub step %c1_i32 iter_args(%o = %init) -> (tensor<1x64xi32>) : i32 {
    %inner:2 = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%idx = %init, %acc = %o)
        -> (tensor<1x64xi32>, tensor<1x64xi32>) : i32 {
      %rem = arith.remsi %idx, %cst : tensor<1x64xi32>
      %sum = arith.addi %acc, %rem : tensor<1x64xi32>
      %l = tt.load %ptr : !tt.ptr<i32>
      %scaled = arith.muli %l, %c128_i32 : i32
      %advance = arith.subi %scaled, %c64_i32 : i32
      %splat = tt.splat %advance : i32 -> tensor<1x64xi32>
      %next = arith.addi %idx, %splat : tensor<1x64xi32>
      scf.yield %next, %sum : tensor<1x64xi32>, tensor<1x64xi32>
    }
    scf.yield %inner#1 : tensor<1x64xi32>
  }
  tt.return %outer : tensor<1x64xi32>
}
// CHECK-LABEL: @nested_loops
// CHECK:         %[[OUTER:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[OFLAG:.*]] = %{{.*}})
// CHECK:           %[[INNER:.*]]:3 = scf.for {{.*}} iter_args(%[[IDX:.*]] = %{{.*}}, %{{.*}} = %{{.*}}, %[[IFLAG:.*]] = %{{.*}})
// CHECK:             %[[COND:.*]] = arith.cmpi sge, %[[IDX]], %{{.*}} : tensor<1x64xi32>
// CHECK:             arith.remui %[[IDX]], %{{.*}} : tensor<1x64xi32>
// CHECK:             %[[IAND:.*]] = arith.andi %[[IFLAG]], %[[COND]] : tensor<1x64xi1>
// CHECK:             scf.yield %{{.*}}, %{{.*}}, %[[IAND]]
// CHECK:           }
// CHECK:           %[[OAND:.*]] = arith.andi %[[OFLAG]], %[[INNER]]#2 : tensor<1x64xi1>
// CHECK:           scf.yield %[[INNER]]#1, %[[OAND]]
// CHECK:         }
// CHECK:         tt.assert %[[OUTER]]#1
}

// -----

// COM: The remainder is nested in an scf.if, which cannot carry an accumulator
// COM: the way an scf.for can, but no loop encloses it, so the assertion stays
// COM: inside the conditional at no cost. Pins that the bail-out condition is
// COM: "the assertion would land inside a loop", not "inside a region".
// COM:
// COM: The checked fact is defined outside the conditional, but the check is
// COM: emitted at the division, so a launch that takes the other branch is not
// COM: aborted for a fact no division depended on.
module {
tt.func public @conditional_outside_loop(%c: i1, %arg0: i32) -> tensor<1x64xi32> {
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %c64_i32 = arith.constant 64 : i32
  %scaled = arith.muli %arg0, %c64_i32 : i32
  %off = tt.splat %scaled : i32 -> tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %exp = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %init = arith.addi %exp, %off : tensor<1x64xi32>
  %res = scf.if %c -> (tensor<1x64xi32>) {
    %rem = arith.remsi %init, %cst : tensor<1x64xi32>
    scf.yield %rem : tensor<1x64xi32>
  } else {
    scf.yield %init : tensor<1x64xi32>
  }
  tt.return %res : tensor<1x64xi32>
}
// CHECK-LABEL: @conditional_outside_loop
// CHECK-SAME:    (%{{.*}}: i1, %[[ARG1:.*]]: i32)
// CHECK:         %[[IDX:.*]] = arith.addi
// CHECK:         scf.if
// CHECK:           %[[COND:.*]] = arith.cmpi sge, %[[ARG1]], %{{.*}} : i32
// CHECK:           tt.assert %[[COND]]
// CHECK:           arith.remui %[[IDX]], %{{.*}} : tensor<1x64xi32>
}

// -----

// COM: The fact the dividend reduces to is an i1, reached through the
// COM: arith.extsi rule. The check is built in the fact's own type, and a signed
// COM: comparison there is exact: `%c` sign-extends to -128 when set, so
// COM: `sge 0` on one bit holds exactly when the dividend is non-negative.
module {
tt.func public @narrow_fact(%arg0: i32) -> tensor<128xi32> {
  %cst = arith.constant dense<128> : tensor<128xi32>
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c = arith.cmpi sgt, %arg0, %c0_i32 : i32
  %ext = arith.extsi %c : i1 to i32
  %scaled = arith.muli %ext, %c128_i32 : i32
  %off = tt.splat %scaled : i32 -> tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %idx = arith.addi %range, %off : tensor<128xi32>
  %rem = arith.remsi %idx, %cst : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @narrow_fact
// CHECK:         %[[C:.*]] = arith.cmpi sgt
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         %[[COND:.*]] = arith.cmpi sge, %[[C]], %[[FALSE]] : i1
// CHECK:         tt.assert %[[COND]]
// CHECK:         arith.remui
}

//===----------------------------------------------------------------------===//
// Negative tests - no deduction to recover, converting would be unsound, or the
// assertion cannot be kept out of a loop body.
//===----------------------------------------------------------------------===//

// -----

// COM: Same shape as @loop_carried_dividend, except the remainder is nested in
// COM: an scf.if. An scf.if cannot carry the accumulator, so the assertion would
// COM: have to stay in the loop body - which costs several times the kernel
// COM: runtime, more than the deduction recovers. Leave the operation signed.
module {
tt.func public @dividend_in_conditional(%c: i1, %ub: i32, %ptr: !tt.ptr<i32>) -> tensor<1x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst = arith.constant dense<128> : tensor<1x64xi32>
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %init = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %res:2 = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%idx = %init, %acc = %init)
      -> (tensor<1x64xi32>, tensor<1x64xi32>) : i32 {
    %sum = scf.if %c -> (tensor<1x64xi32>) {
      %rem = arith.remsi %idx, %cst : tensor<1x64xi32>
      %s = arith.addi %acc, %rem : tensor<1x64xi32>
      scf.yield %s : tensor<1x64xi32>
    } else {
      scf.yield %acc : tensor<1x64xi32>
    }
    %l = tt.load %ptr : !tt.ptr<i32>
    %scaled = arith.muli %l, %c128_i32 : i32
    %advance = arith.subi %scaled, %c64_i32 : i32
    %splat = tt.splat %advance : i32 -> tensor<1x64xi32>
    %next = arith.addi %idx, %splat : tensor<1x64xi32>
    scf.yield %next, %sum : tensor<1x64xi32>, tensor<1x64xi32>
  }
  tt.return %res#1 : tensor<1x64xi32>
}
// CHECK-LABEL: @dividend_in_conditional
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
}

// -----

// COM: A loaded dividend has contiguity and divisibility 1, so the deduction
// COM: could not have concluded anything. This is the `test_bin_op` shape: a
// COM: frontend-style unconditional check would abort here for no benefit.
module {
tt.func public @loaded_dividend(%arg0: tensor<128x!tt.ptr<i32>>) -> (tensor<128xi32>, tensor<128xi32>) {
  %cst = arith.constant dense<4> : tensor<128xi32>
  %x = tt.load %arg0 : tensor<128x!tt.ptr<i32>>
  %rem = arith.remsi %x, %cst : tensor<128xi32>
  %div = arith.divsi %x, %cst : tensor<128xi32>
  tt.return %rem, %div : tensor<128xi32>, tensor<128xi32>
}
// CHECK-LABEL: @loaded_dividend
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
// CHECK:         arith.divsi
}

// -----

// COM: A runtime scalar divisor is constant along the dimension but has no
// COM: constant value, so it cannot be checked for positivity. Deliberately
// COM: left signed: correct, just not optimized.
module {
tt.func public @runtime_divisor(%arg0: i32) -> tensor<128xi32> {
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %d = tt.splat %arg0 : i32 -> tensor<128xi32>
  %rem = arith.remsi %range, %d : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @runtime_divisor
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
}

// -----

// COM: A negative constant divisor satisfies the deduction's match conditions,
// COM: but divui/remui would reinterpret it as a large positive value. Asserting
// COM: the divisor instead would emit a check that fails on every launch.
module {
tt.func public @negative_divisor() -> tensor<128xi32> {
  %cst = arith.constant dense<-4> : tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %div = arith.divsi %range, %cst : tensor<128xi32>
  tt.return %div : tensor<128xi32>
}
// CHECK-LABEL: @negative_divisor
// CHECK-NOT:     tt.assert
// CHECK:         arith.divsi
}

// -----

// COM: Offsetting a range by a runtime scalar drops its divisibility to 1, so
// COM: the deduction's gcd is 1 and there is nothing to recover. Pins the
// COM: `gcd > 1` term: this shape is common in real kernels.
module {
tt.func public @gcd_is_one(%arg0: i32) -> tensor<128xi32> {
  %cst = arith.constant dense<5> : tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %off = tt.splat %arg0 : i32 -> tensor<128xi32>
  %idx = arith.addi %range, %off : tensor<128xi32>
  %rem = arith.remsi %idx, %cst : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @gcd_is_one
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
}

// -----

// COM: The match conditions hold, but every element of the dividend is negative,
// COM: so the assertion would fail on every launch and the unsigned result would
// COM: differ from the signed one. A kernel that computes correctly today must
// COM: not start aborting, so leave it signed. The range is [-256, -129], which
// COM: only the integer range analysis can establish: the guard is a range
// COM: query, not a constant check.
module {
tt.func public @provably_negative_computed_dividend() -> tensor<128xi32> {
  %cst = arith.constant dense<-256> : tensor<128xi32>
  %cst16 = arith.constant dense<16> : tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %neg = arith.addi %range, %cst : tensor<128xi32>
  %div = arith.divsi %neg, %cst16 : tensor<128xi32>
  tt.return %div : tensor<128xi32>
}
// CHECK-LABEL: @provably_negative_computed_dividend
// CHECK-NOT:     tt.assert
// CHECK:         arith.divsi
}

// -----

// COM: Two independent runtime offsets, so the dividend is non-negative only if
// COM: both kernel arguments are. A conjunction cannot be partially satisfied, so
// COM: a dividend needing more than one fact is left signed rather than guarded
// COM: by a pile of checks.
module {
tt.func public @dividend_needs_two_facts(%arg0: i32, %arg1: i32) -> tensor<128xi32> {
  %cst = arith.constant dense<128> : tensor<128xi32>
  %c64_i32 = arith.constant 64 : i32
  %s0 = arith.muli %arg0, %c64_i32 : i32
  %s1 = arith.muli %arg1, %c64_i32 : i32
  %o0 = tt.splat %s0 : i32 -> tensor<128xi32>
  %o1 = tt.splat %s1 : i32 -> tensor<128xi32>
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %t = arith.addi %range, %o0 : tensor<128xi32>
  %idx = arith.addi %t, %o1 : tensor<128xi32>
  %rem = arith.remsi %idx, %cst : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @dividend_needs_two_facts
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
}

// -----

// COM: Both remaining sites bail when the result is not a ranked tensor, so a
// COM: scalar operation never matches.
module {
tt.func public @scalar_operands(%arg0: i32) -> i32 {
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %arg0, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @scalar_operands
// CHECK-NOT:     tt.assert
// CHECK:         arith.remsi
}
