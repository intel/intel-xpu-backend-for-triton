// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-range-analysis -verify-diagnostics=only-expected | FileCheck %s

// COM: An `llvm.intr.assume` constrains a value at a use only if it is certain
// COM: to execute whenever that use does. One section per class
// COM: the applicability helper must reject, plus a control that must keep
// COM: working. `%use` is an identity add, so its range is exactly the range
// COM: the analysis holds for `%n`: the full range means the fact did not
// COM: reach the use, `[64, ...]` means it did.

// COM: The comparison is hoisted above the scf.if; the assume stays inside.
// CHECK-LABEL: tt.func @assume_scoped_to_branch
module {
  tt.func @assume_scoped_to_branch(%n: i32, %flag: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    scf.if %flag {
      llvm.intr.assume %cmp : i1
      scf.yield
    }
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    tt.return
  }
}

// -----

// COM: The assume sits after the loop; the use is inside it. A parent block
// COM: dominates the loop body, but the loop may never terminate.
// CHECK-LABEL: tt.func @assume_after_loop_does_not_reach_inside
module {
  tt.func @assume_after_loop_does_not_reach_inside(%n: i32, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    scf.for %i = %c0 to %ub step %c1 : i32 {
      // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
      %use = arith.addi %n, %c0 : i32
      scf.yield
    }
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A non-terminating scf.while sits between the use and the assume.
// CHECK-LABEL: tt.func @non_terminating_region_blocks_backward_assume
module {
  tt.func @non_terminating_region_blocks_backward_assume(%n: i32, %flag: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.while () : () -> () {
      scf.condition(%flag)
    } do {
      scf.yield
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: tt.atomic_poll without a timeout polls until the expected value
// COM: appears, so it may spin forever.
// CHECK-LABEL: tt.func @no_timeout_poll_blocks_backward_assume
module {
  tt.func @no_timeout_poll_blocks_backward_assume(%n: i32, %p: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %ok = tt.atomic_poll acquire, gpu, %p, %c0 : !tt.ptr<i32>, i32 -> i1
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: An impure tt.extern_elementwise lowers to an external call that may
// COM: never return.
// CHECK-LABEL: tt.func @impure_extern_call_blocks_backward_assume
module {
  tt.func @impure_extern_call_blocks_backward_assume(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %e = tt.extern_elementwise %n {libname = "l", libpath = "p", symbol = "s", pure = false} : (i32) -> i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: An impure tt.elementwise_inline_asm is opaque for the same reason.
// CHECK-LABEL: tt.func @impure_inline_asm_blocks_backward_assume
module {
  tt.func @impure_inline_asm_blocks_backward_assume(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %a = tt.elementwise_inline_asm "nop" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %n : i32 -> i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: Control: a straight-line assume after the use, with nothing in between
// COM: that can abort or spin, DOES apply. The existing range tests rely on
// COM: this, so the fix must not reject it.
// CHECK-LABEL: tt.func @straight_line_backward_assume_applies
module {
  tt.func @straight_line_backward_assume_applies(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}
