// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-range-analysis -verify-diagnostics=only-expected | FileCheck %s

// COM: Policy (#8368): for an integer type wider than 64 bits the analysis
// COM: skips assume facts and trip counts (they narrow to int64_t), even when
// COM: the numbers are small; constants and arithmetic on wide values are still
// COM: analyzed. Before the fix, the first three cases aborted or gave a wrong
// COM: range.

// COM: 1025 = kDefaultMaxTripCount + 1, the total trip count reported for a
// COM: loop whose trip count is unknown (getTripCount has no value).
// CHECK-LABEL: tt.func @loop_unknown_bound_i128
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @loop_unknown_bound_i128(%n: i128) {
    %c0 = arith.constant 0 : i128
    %c1 = arith.constant 1 : i128
    // expected-remark@+1 {{inferred total trip count: 1025}}
    scf.for %iv = %c0 to %n step %c1 : i128 {
      scf.yield
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func @assume_big_constant_i128
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 340282366920938463463374607431768211455] signed : [-170141183460469231731687303715884105728, 170141183460469231731687303715884105727]}}
  tt.func @assume_big_constant_i128(%a: i128) {
    %big = arith.constant 18446744073709551616 : i128
    %le = arith.cmpi sle, %a, %big : i128
    llvm.intr.assume %le : i1
    tt.return
  }
}

// -----

// COM: `a ule -1` is vacuous in i128; the wrong range was [0, 2^64-1].
// CHECK-LABEL: tt.func @assume_ule_minus_one_i128
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 340282366920938463463374607431768211455] signed : [-170141183460469231731687303715884105728, 170141183460469231731687303715884105727]}}
  tt.func @assume_ule_minus_one_i128(%a: i128) {
    %m1 = arith.constant -1 : i128
    %le = arith.cmpi ule, %a, %m1 : i128
    llvm.intr.assume %le : i1
    tt.return
  }
}

// -----

// COM: Boundary (assume): exactly 64 bits is still analyzed.
// CHECK-LABEL: tt.func @assume_i64_still_analyzed
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 100]}}
  tt.func @assume_i64_still_analyzed(%a: i64) {
    %c = arith.constant 100 : i64
    %le = arith.cmpi sle, %a, %c : i64
    llvm.intr.assume %le : i1
    tt.return
  }
}

// -----

// COM: Policy: 100 fits int64_t, yet `a` must stay unconstrained. A future
// COM: wide-type feature must flip this on purpose.
// CHECK-LABEL: tt.func @assume_small_i128
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 340282366920938463463374607431768211455] signed : [-170141183460469231731687303715884105728, 170141183460469231731687303715884105727]}}
  tt.func @assume_small_i128(%a: i128) {
    %c100 = arith.constant 100 : i128
    %cmp = arith.cmpi sle, %a, %c100 : i128
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: Same policy for loops: bounds 0..10 fit int64_t, yet the trip count is
// COM: unknown (1025, as in the first case), so the loop-carried result gets
// COM: the full range.
// CHECK-LABEL: tt.func @loop_small_i128
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @loop_small_i128() {
    %c0 = arith.constant 0 : i128
    %c10 = arith.constant 10 : i128
    %c1 = arith.constant 1 : i128
    // expected-remark@+2 {{unsigned : [0, 340282366920938463463374607431768211455] signed : [-170141183460469231731687303715884105728, 170141183460469231731687303715884105727]}}
    // expected-remark@+1 {{inferred total trip count: 1025}}
    %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (i128) : i128 {
      %n = arith.addi %acc, %c1 : i128
      scf.yield %n : i128
    }
    tt.return
  }
}

// -----

// COM: Boundary (loops): exactly 64 bits is still analyzed. 8 iterations, not
// COM: 10: diagnostics match by substring and "10" is a prefix of 1025.
// CHECK-LABEL: tt.func @loop_small_i64
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @loop_small_i64() {
    %c0 = arith.constant 0 : i64
    %c8 = arith.constant 8 : i64
    %c1 = arith.constant 1 : i64
    // expected-remark@+2 {{unsigned : [1, 8] signed : [1, 8]}}
    // expected-remark@+1 {{inferred total trip count: 8}}
    %r = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
      %n = arith.addi %acc, %c1 : i64
      scf.yield %n : i64
    }
    tt.return
  }
}

// -----

// COM: i65 is the smallest wide type: 2^63 needs 65 signed bits.
// CHECK-LABEL: tt.func @assume_big_constant_i65
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 36893488147419103231] signed : [-18446744073709551616, 18446744073709551615]}}
  tt.func @assume_big_constant_i65(%a: i65) {
    %big = arith.constant 9223372036854775808 : i65
    %le = arith.cmpi sle, %a, %big : i65
    llvm.intr.assume %le : i1
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func @loop_unknown_bound_i65
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @loop_unknown_bound_i65(%n: i65) {
    %c0 = arith.constant 0 : i65
    %c1 = arith.constant 1 : i65
    // expected-remark@+1 {{inferred total trip count: 1025}}
    scf.for %iv = %c0 to %n step %c1 : i65 {
      scf.yield
    }
    tt.return
  }
}
