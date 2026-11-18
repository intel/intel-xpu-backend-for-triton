// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: Two sibling versionable loops in the same function. The versioner erases
// COM: the loop it versions, so the walk must not descend into it, but it must
// COM: still reach the loops following it: check both loops are versioned.
tt.func public @two_sibling_loops(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) -> (tensor<32xf16>, tensor<32xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %p0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %p = tt.addptr %p0, %r : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %n = arith.addi %arg1, %c31_i32 : i32
  %ub = arith.divsi %n, %c32_i32 : i32
  %res1 = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %cst) -> (tensor<32xf16>)  : i32 {
    %m0 = arith.muli %iv, %c32_i32 : i32
    %m1 = arith.subi %arg1, %m0 : i32
    %m2 = tt.splat %m1 : i32 -> tensor<32xi32>
    %m = arith.cmpi slt, %r, %m2 : tensor<32xi32>
    %l = tt.load %p, %m, %cst : tensor<32x!tt.ptr<f16>>
    %a = arith.addf %acc, %l : tensor<32xf16>
    scf.yield %a : tensor<32xf16>
  }
  %res2 = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %cst) -> (tensor<32xf16>)  : i32 {
    %m0 = arith.muli %iv, %c32_i32 : i32
    %m1 = arith.subi %arg1, %m0 : i32
    %m2 = tt.splat %m1 : i32 -> tensor<32xi32>
    %m = arith.cmpi slt, %r, %m2 : tensor<32xi32>
    %l = tt.load %p, %m, %cst : tensor<32x!tt.ptr<f16>>
    %a = arith.addf %acc, %l : tensor<32xf16>
    scf.yield %a : tensor<32xf16>
  }
  tt.return %res1, %res2 : tensor<32xf16>, tensor<32xf16>
}

// CHECK-LABEL: @two_sibling_loops
// COM: First loop versioned: mask free load in the "then" region.
// CHECK:       [[IF1:%[0-9]+]] = scf.if {{%[0-9]+}} -> (tensor<32xf16>) {
// CHECK:         scf.for
// CHECK:           tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
// CHECK:       } else {
// CHECK:         scf.for
// CHECK:           tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f16>>
// CHECK:       }
// COM: Second loop versioned too.
// CHECK:       [[IF2:%[0-9]+]] = scf.if {{%[0-9]+}} -> (tensor<32xf16>) {
// CHECK:         scf.for
// CHECK:           tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
// CHECK:       } else {
// CHECK:         scf.for
// CHECK:           tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f16>>
// CHECK:       }
// CHECK:       tt.return [[IF1]], [[IF2]] : tensor<32xf16>, tensor<32xf16>
