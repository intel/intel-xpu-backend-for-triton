// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

// COM: A loop invariant mask in boundary check shape
// COM: `(splat(offset) + make_range(0, END)) cmp dense<constant>` but with an
// COM: `eq` predicate. Such a mask is satisfied by a single lane, so no scalar
// COM: condition on the offset implies it: the loop must not be versioned.
tt.func public @eq_boundary_shape(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %bound = arith.constant dense<128> : tensor<32xi32>
  %rng = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %off = tt.splat %arg1 : i32 -> tensor<32xi32>
  %offs = arith.addi %off, %rng : tensor<32xi32>
  %mask = arith.cmpi eq, %offs, %bound : tensor<32xi32>
  %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %ptrs = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %l = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %n = arith.addf %acc, %l : tensor<32xf32>
    scf.yield %n : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @eq_boundary_shape
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f32>>
// CHECK-NOT:   scf.if

// -----

// COM: Same as above with a `ne` predicate.
tt.func public @ne_boundary_shape(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %bound = arith.constant dense<128> : tensor<32xi32>
  %rng = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %off = tt.splat %arg1 : i32 -> tensor<32xi32>
  %offs = arith.addi %off, %rng : tensor<32xi32>
  %mask = arith.cmpi ne, %offs, %bound : tensor<32xi32>
  %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %ptrs = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %l = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %n = arith.addf %acc, %l : tensor<32xf32>
    scf.yield %n : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @ne_boundary_shape
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f32>>
// CHECK-NOT:   scf.if
