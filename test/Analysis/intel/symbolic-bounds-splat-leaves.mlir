// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-intel-symbolic-bounds="splat-leaves=1" -verify-diagnostics=only-expected | FileCheck %s

// COM: With `splat-leaves`, each comparison reports `rhs - lhs` as the prover
// COM: normalizes it when a splatted scalar is kept only if exact, and the
// COM: wrap obligations left over. A row-limit mask reads as `H - lane`.

// CHECK-LABEL: tt.func @row_limit
module {
  tt.func @row_limit(%base: i32, %bound: i32) {
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32> loc("lane")
    %bs = tt.splat %base : i32 -> tensor<64xi32>
    %rows = arith.addi %bs, %lane : tensor<64xi32>
    %ns = tt.splat %bound : i32 -> tensor<64xi32>
    // expected-remark@+1 {{diff: -arg0 + arg1 - lane; wraps: [arith.addi]}}
    %mask = arith.cmpi slt, %rows, %ns : tensor<64xi32>
    tt.return
  }
}

// -----

// COM: The splatted bound is scalar arithmetic that can wrap, so it stays the
// COM: kernel's own value, and its wrap obligation is dropped.

// CHECK-LABEL: tt.func @k_tail
module {
  tt.func @k_tail(%kt: i32, %ko: i32) {
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32> loc("lane")
    %kr = arith.subi %kt, %ko : i32 loc("kr")
    %ks = tt.splat %kr : i32 -> tensor<64xi32>
    // expected-remark@+1 {{diff: -lane + opaque(kr); wraps: []}}
    %mask = arith.cmpi slt, %lane, %ks : tensor<64xi32>
    tt.return
  }
}

// -----

// COM: Inductor's order, `off + arange(0, N)[:, None]`: expand before the add.

// CHECK-LABEL: tt.func @expand_before_add
module {
  tt.func @expand_before_add(%off: i32, %M: i32) {
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32> loc("lane")
    %col = tt.expand_dims %lane {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %os = tt.splat %off : i32 -> tensor<64x1xi32>
    %rows = arith.addi %os, %col : tensor<64x1xi32>
    %ms = tt.splat %M : i32 -> tensor<64x1xi32>
    // expected-remark@+1 {{diff: -arg0 + arg1 - lane; wraps: [arith.addi]}}
    %mask = arith.cmpi slt, %rows, %ms : tensor<64x1xi32>
    tt.return
  }
}

// -----

// COM: `pid * 64` can wrap, so the row base is the kernel's product itself.

// CHECK-LABEL: tt.func @pid_base
module {
  tt.func @pid_base(%M: i32) {
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %off = arith.muli %pid, %c64 : i32 loc("off")
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32> loc("lane")
    %os = tt.splat %off : i32 -> tensor<64xi32>
    %rows = arith.addi %os, %lane : tensor<64xi32>
    %ms = tt.splat %M : i32 -> tensor<64xi32>
    // expected-remark@+1 {{diff: arg0 - lane - opaque(off); wraps: [arith.addi]}}
    %mask = arith.cmpi slt, %rows, %ms : tensor<64xi32>
    tt.return
  }
}
