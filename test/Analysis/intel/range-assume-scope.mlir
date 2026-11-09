// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-range-analysis -verify-diagnostics=only-expected | FileCheck %s

// COM: An `llvm.intr.assume` constrains a value at a use only if it is certain
// COM: to execute whenever that use does. One section per class
// COM: the applicability helper must reject, plus a control that must keep
// COM: working. `%use` is an identity add, so its range is exactly the range
// COM: the analysis holds for `%n`: the full range means the fact did not
// COM: reach the use, `[64, ...]` means it did. The last sections pin the
// COM: range of a value the analysis does not model, which holds for every
// COM: use, so the assume must be certain to execute once the value exists.

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
// COM: dominates the loop body, but loops are not modelled, so a loop between
// COM: the use and the assume blocks it.
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
// COM: appears, so it may spin forever. A timeout does not help: it is a
// COM: runtime value and may be unbounded.
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

// -----

// COM: tt.assert aborts when its condition is false.
// CHECK-LABEL: tt.func @assert_blocks_backward_assume
module {
  tt.func @assert_blocks_backward_assume(%n: i32, %ok: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    tt.assert %ok, "must hold" : i1
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A call may not return.
// CHECK-LABEL: tt.func @call_blocks_backward_assume
module {
  tt.func private @callee() {
    tt.return
  }
  tt.func @call_blocks_backward_assume(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    tt.call @callee() : () -> ()
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A loop before `%v` is defined has finished whenever `%v` exists, so it
// COM: does not stop the assume from applying to `%v`.
// CHECK-LABEL: tt.func @loop_before_definition_does_not_block
module {
  tt.func @loop_before_definition_does_not_block(%p: !tt.ptr<i32>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.yield
    }
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %v = tt.load %p : !tt.ptr<i32>
    %cmp = arith.cmpi sge, %v, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A loop between `%v` and the assume, which uses `%v`, still stops the
// COM: assume from applying: loops are not modelled.
// CHECK-LABEL: tt.func @loop_after_definition_blocks
module {
  tt.func @loop_after_definition_blocks(%p: !tt.ptr<i32>, %q: !tt.ptr<i32>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %v = tt.load %p : !tt.ptr<i32>
    scf.for %i = %c0 to %ub step %c1 : i32 {
      tt.store %q, %v : !tt.ptr<i32>
      scf.yield
    }
    %cmp = arith.cmpi sge, %v, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: The region of a tt.reduce has run by the time its result exists.
// CHECK-LABEL: tt.func @reduce_result_takes_assume
module {
  tt.func @reduce_result_takes_assume() {
    %c64 = arith.constant 64 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.maxsi %a, %b : i32
      tt.reduce.return %s : i32
    }) : (tensor<4xi32>) -> i32
    %cmp = arith.cmpi sge, %r, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: An scf.if whose body always finishes does not stop a later assume.
// CHECK-LABEL: tt.func @scf_if_between_does_not_block
module {
  tt.func @scf_if_between_does_not_block(%n: i32, %flag: i1, %q: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      tt.store %q, %n : !tt.ptr<i32>
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: Same with both branches.
// CHECK-LABEL: tt.func @scf_if_else_between_does_not_block
module {
  tt.func @scf_if_else_between_does_not_block(%n: i32, %flag: i1, %q: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      tt.store %q, %n : !tt.ptr<i32>
    } else {
      tt.store %q, %c0 : !tt.ptr<i32>
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: An scf.if nested in another, both finishing.
// CHECK-LABEL: tt.func @nested_if_in_if_does_not_block
module {
  tt.func @nested_if_in_if_does_not_block(%n: i32, %flag: i1, %q: !tt.ptr<i32>, %flag2: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      scf.if %flag2 {
        tt.store %q, %n : !tt.ptr<i32>
      }
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A tt.reduce between the use and the assume finishes.
// CHECK-LABEL: tt.func @reduce_between_does_not_block
module {
  tt.func @reduce_between_does_not_block(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.maxsi %a, %b : i32
      tt.reduce.return %s : i32
    }) : (tensor<4xi32>) -> i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A tt.scan between the use and the assume finishes.
// CHECK-LABEL: tt.func @scan_between_does_not_block
module {
  tt.func @scan_between_does_not_block(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.scan.return %s : i32
    }) : (tensor<4xi32>) -> tensor<4xi32>
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A tt.assert inside the scf.if may abort, so the scf.if blocks.
// CHECK-LABEL: tt.func @assert_in_then_blocks
module {
  tt.func @assert_in_then_blocks(%n: i32, %flag: i1, %q: !tt.ptr<i32>, %ok: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      tt.assert %ok, "must hold" : i1
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: Same, with the tt.assert in the else branch.
// CHECK-LABEL: tt.func @assert_in_else_blocks
module {
  tt.func @assert_in_else_blocks(%n: i32, %flag: i1, %q: !tt.ptr<i32>, %ok: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      tt.store %q, %n : !tt.ptr<i32>
    } else {
      tt.assert %ok, "must hold" : i1
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A loop nested in the scf.if blocks: loops are not modelled.
// CHECK-LABEL: tt.func @loop_in_if_blocks
module {
  tt.func @loop_in_if_blocks(%n: i32, %flag: i1, %q: !tt.ptr<i32>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %c1 = arith.constant 1 : i32
    scf.if %flag {
      scf.for %i = %c0 to %ub step %c1 : i32 {
        tt.store %q, %n : !tt.ptr<i32>
      }
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A tt.assert two scf.if levels down still blocks.
// CHECK-LABEL: tt.func @assert_in_if_in_if_blocks
module {
  tt.func @assert_in_if_in_if_blocks(%n: i32, %flag: i1, %q: !tt.ptr<i32>, %flag2: i1, %ok: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    scf.if %flag {
      scf.if %flag2 {
        tt.assert %ok, "must hold" : i1
      }
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: tt.map_elementwise blocks even with a pure body: its region may hold
// COM: a CFG cycle.
// CHECK-LABEL: tt.func @map_elementwise_blocks
module {
  tt.func @map_elementwise_blocks(%n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %m = "tt.map_elementwise"(%x) <{pack = 1 : i32}> ({
    ^bb0(%a: i32):
      tt.map_elementwise.return %a : i32
    }) : (tensor<4xi32>) -> tensor<4xi32>
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: Same with a timeout.
// CHECK-LABEL: tt.func @timeout_poll_blocks_backward_assume
module {
  tt.func @timeout_poll_blocks_backward_assume(%n: i32, %p: !tt.ptr<i32>, %t: i64) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %ok = tt.atomic_poll acquire, gpu, %p, %c0 timeout %t : !tt.ptr<i32>, i32 -> i1
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A loop in a tt.reduce combiner blocks the tt.reduce.
// CHECK-LABEL: tt.func @loop_in_reduce_blocks
module {
  tt.func @loop_in_reduce_blocks(%n: i32, %flag: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      scf.while () : () -> () {
        scf.condition(%flag)
      } do {
        scf.yield
      }
      %s = arith.maxsi %a, %b : i32
      tt.reduce.return %s : i32
    }) : (tensor<4xi32>) -> i32
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A loop in a tt.scan combiner blocks the tt.scan.
// CHECK-LABEL: tt.func @loop_in_scan_blocks
module {
  tt.func @loop_in_scan_blocks(%n: i32, %flag: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %use = arith.addi %n, %c0 : i32
    %x = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      scf.while () : () -> () {
        scf.condition(%flag)
      } do {
        scf.yield
      }
      %s = arith.addi %a, %b : i32
      tt.scan.return %s : i32
    }) : (tensor<4xi32>) -> tensor<4xi32>
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}

// -----

// COM: A use inside an scf.if that always finishes takes a later assume.
// CHECK-LABEL: tt.func @use_inside_scf_if_takes_later_assume
module {
  tt.func @use_inside_scf_if_takes_later_assume(%n: i32, %flag: i1, %q: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    scf.if %flag {
      // expected-remark@+1 {{unsigned : [64, 2147483647] signed : [64, 2147483647]}}
      %use = arith.addi %n, %c0 : i32
      tt.store %q, %use : !tt.ptr<i32>
    }
    %cmp = arith.cmpi sge, %n, %c64 : i32
    llvm.intr.assume %cmp : i1
    tt.return
  }
}
