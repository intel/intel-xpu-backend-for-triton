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
