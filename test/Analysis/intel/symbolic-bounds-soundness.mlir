// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-symbolic-bounds -verify-diagnostics=only-expected | FileCheck %s

// COM: Soundness regressions: each comparison below is false at the input its
// COM: comment names, so a verdict must never be Satisfied, and a Conditional
// COM: must exclude that input.

// COM: x + 1 > x with x an unconstrained i8 load: false at x = 127.
// CHECK-LABEL: tt.func @unbounded_varying_cancels
module {
  tt.func @unbounded_varying_cancels(%p: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c1_i8 = arith.constant 1 : i8
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %x = tt.load %p : !tt.ptr<i8>
      %y = arith.addi %x, %c1_i8 : i8
      // expected-remark@+1 {{verdict: Unknown}}
      %cmp = arith.cmpi sgt, %y, %x : i8
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: iv - lb >= 0 over lb to ub step 2^62: at lb = INT64_MIN, ub = 1 the
// COM: iv = 0 iteration computes 2^63, which wraps negative.
// CHECK-LABEL: tt.func @i64_guard_fold_overflow
module {
  tt.func @i64_guard_fold_overflow(%lb: i64, %ub: i64) {
    %c0 = arith.constant 0 : i64
    %step = arith.constant 4611686018427387904 : i64
    scf.for %iv = %lb to %ub step %step : i64 {
      %d = arith.subi %iv, %lb : i64
      // expected-remark@+1 {{verdict: Unknown}}
      %cmp = arith.cmpi sge, %d, %c0 : i64
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: iv + lane < n + m over 0 to n step 4, n and m sign-extended from i8:
// COM: at n = 5, m = 0 the iv = 4 iteration has lanes 5..7 false.
// CHECK-LABEL: tt.func @improving_candidate_condition
module {
  tt.func @improving_candidate_condition(%n8: i8, %m8: i8) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %n = arith.extsi %n8 : i8 to i32
    %m = arith.extsi %m8 : i8 to i32
    %nm = arith.addi %n, %m : i32
    %nms = tt.splat %nm : i32 -> tensor<4xi32>
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    scf.for %iv = %c0 to %n step %c4 : i32 {
      %ivs = tt.splat %iv : i32 -> tensor<4xi32>
      %idx = arith.addi %ivs, %lane : tensor<4xi32>
      // expected-remark@+1 {{verdict: Conditional{arg0 divisible by 4; arg1 >= 0}}}
      %cmp = arith.cmpi slt, %idx, %nms : tensor<4xi32>
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: i8 4*(x/4) + 4 > x: at x = 124 the sum is 128, which wraps to -128.
// CHECK-LABEL: tt.func @quotient_upper_guard
module {
  tt.func @quotient_upper_guard(%x: i8) {
    %c4 = arith.constant 4 : i8
    %q = arith.divsi %x, %c4 : i8
    %t = arith.muli %q, %c4 : i8
    %y = arith.addi %t, %c4 : i8
    // expected-remark@+1 {{verdict: Conditional{arg0 >= 0; arg0 <= 123}}}
    %cmp = arith.cmpi sgt, %y, %x : i8
    tt.return
  }
}

// -----

// COM: q = range(0, 4) / 2 broadcast along both axes: 2*q_col <= 2*q_row + 1 is
// COM: false at q_col = 1, q_row = 0.
// CHECK-LABEL: tt.func @quotient_placement
module {
  tt.func @quotient_placement() {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %c2v = arith.constant dense<2> : tensor<4xi32>
    %q = arith.divsi %r, %c2v : tensor<4xi32>
    %qc = tt.expand_dims %q {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %qr = tt.expand_dims %q {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %qcb = tt.broadcast %qc : tensor<4x1xi32> -> tensor<4x4xi32>
    %qrb = tt.broadcast %qr : tensor<1x4xi32> -> tensor<4x4xi32>
    %c2t = arith.constant dense<2> : tensor<4x4xi32>
    %c1t = arith.constant dense<1> : tensor<4x4xi32>
    %lhs = arith.muli %qcb, %c2t : tensor<4x4xi32>
    %rhs0 = arith.muli %qrb, %c2t : tensor<4x4xi32>
    %rhs = arith.addi %rhs0, %c1t : tensor<4x4xi32>
    // expected-remark@+1 {{verdict: Unknown}}
    %cmp = arith.cmpi sle, %lhs, %rhs : tensor<4x4xi32>
    tt.return
  }
}

// -----

// COM: assume(X == 12), q = X / 4 = 3, iv + lane < q over 0 to q step 2 with
// COM: lanes [0, 2): the iv = 2 iteration has lane 1 at 3 < 3, false.
// CHECK-LABEL: tt.func @quotient_range_pruning
module {
  tt.func @quotient_range_pruning(%X: i32) {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c12 = arith.constant 12 : i32
    %eq = arith.cmpi eq, %X, %c12 : i32
    llvm.intr.assume %eq : i1
    %q = arith.divsi %X, %c4 : i32
    %lane = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %qs = tt.splat %q : i32 -> tensor<2xi32>
    scf.for %iv = %c0 to %q step %c2 : i32 {
      %ivs = tt.splat %iv : i32 -> tensor<2xi32>
      %idx = arith.addi %ivs, %lane : tensor<2xi32>
      // expected-remark@+1 {{verdict: Conditional{(arg0 div 4) divisible by 2}}}
      %cmp = arith.cmpi slt, %idx, %qs : tensor<2xi32>
      scf.yield
    }
    tt.return
  }
}

// -----

// COM: i64 (0 - x) + x >= 0: the lower wrap guard of 0 - x is -x >= INT64_MIN,
// COM: whose division by -1 needs 2^63. It holds for every x, so only the
// COM: upper guard (x != INT64_MIN) remains.
// CHECK-LABEL: tt.func @neg_wrap_int64_min_bound
module {
  tt.func @neg_wrap_int64_min_bound(%x: i64) {
    %c0 = arith.constant 0 : i64
    %n = arith.subi %c0, %x : i64
    %s = arith.addi %n, %x : i64
    // expected-remark@+1 {{verdict: Conditional{arg0 >= -9223372036854775807}}}
    %cmp = arith.cmpi sge, %s, %c0 : i64
    tt.return
  }
}

// -----

// COM: i128 is wider than the int64_t domain the prover reasons in. Read as its
// COM: low 64 bits an unconstrained i128 looks non-negative, so this must be
// COM: Unknown rather than Satisfied.
// CHECK-LABEL: tt.func @i128_unconstrained_nonneg
module {
  tt.func @i128_unconstrained_nonneg(%a: i128) {
    %c0 = arith.constant 0 : i128
    // expected-remark@+1 {{verdict: Unknown}}
    %cmp = arith.cmpi sge, %a, %c0 : i128
    tt.return
  }
}

// -----

// COM: A constant of 2^64 reads as 0 in its low 64 bits, which would make
// COM: `2^64 <= 0` true.
// CHECK-LABEL: tt.func @i128_constant_beyond_int64
module {
  tt.func @i128_constant_beyond_int64() {
    %big = arith.constant 18446744073709551616 : i128
    %c0 = arith.constant 0 : i128
    // expected-remark@+1 {{verdict: Unknown}}
    %cmp = arith.cmpi sle, %big, %c0 : i128
    tt.return
  }
}
