// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: A canonically masked load whose `N` is a kernel argument (no defining
// COM: operation), in a loop with a constant upper bound. A constant upper bound
// COM: can only be matched against the folded canonical form (N+END-1)/END when
// COM: `N` is a constant too, so the loop must not be versioned. Casting the
// COM: (null) defining operation of `N` used to abort the compiler.
tt.func public @const_ub_dynamic_n(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<32xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %p0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %p = tt.addptr %p0, %r : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %res = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<32xf16>)  : i32 {
    %m0 = arith.muli %iv, %c32_i32 : i32
    %m1 = arith.subi %arg1, %m0 : i32
    %m2 = tt.splat %m1 : i32 -> tensor<32xi32>
    %m = arith.cmpi slt, %r, %m2 : tensor<32xi32>
    %l = tt.load %p, %m, %cst : tensor<32x!tt.ptr<f16>>
    %a = arith.addf %acc, %l : tensor<32xf16>
    scf.yield %a : tensor<32xf16>
  }
  tt.return %res : tensor<32xf16>
}

// CHECK-LABEL: @const_ub_dynamic_n
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f16>>
// CHECK-NOT:   scf.if
