// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

// COM: Two canonically masked loads in one loop with *different* `N` and `END`.
// COM: The loop upper bound is (%arg2 + 31) / 32, so a versioning condition
// COM: derived from it implies load A's mask but not load B's, which depends on
// COM: %arg3 and END=64. Because the versioned loop drops the mask of *every*
// COM: collected operation, the loop must not be versioned at all.
tt.func public @two_canonical_masks(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32) -> (tensor<32xf16>, tensor<64xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %cstA = arith.constant dense<0.000000e+00> : tensor<32xf16>
  %cstB = arith.constant dense<0.000000e+00> : tensor<64xf16>
  %rA = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %rB = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %pA0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %pA = tt.addptr %pA0, %rA : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %pB0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  %pB = tt.addptr %pB0, %rB : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  %n = arith.addi %arg2, %c31_i32 : i32
  %ub = arith.divsi %n, %c32_i32 : i32
  %res:2 = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%accA = %cstA, %accB = %cstB) -> (tensor<32xf16>, tensor<64xf16>)  : i32 {
    // COM: Mask A: range(0,32) < %arg2 - iv*32 (matches the loop upper bound).
    %mA0 = arith.muli %iv, %c32_i32 : i32
    %mA1 = arith.subi %arg2, %mA0 : i32
    %mA2 = tt.splat %mA1 : i32 -> tensor<32xi32>
    %mA = arith.cmpi slt, %rA, %mA2 : tensor<32xi32>
    %lA = tt.load %pA, %mA, %cstA : tensor<32x!tt.ptr<f16>>
    // COM: Mask B: range(0,64) < %arg3 - iv*64 (unrelated to the loop bound).
    %mB0 = arith.muli %iv, %c64_i32 : i32
    %mB1 = arith.subi %arg3, %mB0 : i32
    %mB2 = tt.splat %mB1 : i32 -> tensor<64xi32>
    %mB = arith.cmpi slt, %rB, %mB2 : tensor<64xi32>
    %lB = tt.load %pB, %mB, %cstB : tensor<64x!tt.ptr<f16>>
    %nA = arith.addf %accA, %lA : tensor<32xf16>
    %nB = arith.addf %accB, %lB : tensor<64xf16>
    scf.yield %nA, %nB : tensor<32xf16>, tensor<64xf16>
  }
  tt.return %res#0, %res#1 : tensor<32xf16>, tensor<64xf16>
}

// CHECK-LABEL: @two_canonical_masks
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// COM: Both loads keep their mask.
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst{{.*}} : tensor<32x!tt.ptr<f16>>
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst{{.*}} : tensor<64x!tt.ptr<f16>>
// CHECK-NOT:   scf.if

// -----

// COM: A single canonically masked load whose `N` (%arg3) and `END` (64) do not
// COM: match the loop upper bound's `N` (%arg2) and `END` (32). Recognizing the
// COM: upper bound shape alone is not sufficient: the versioning condition is
// COM: generated from the upper bound, so it does not imply this mask.
tt.func public @ub_mask_mismatch(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32) -> tensor<64xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf16>
  %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %p0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  %p = tt.addptr %p0, %r : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  %n = arith.addi %arg2, %c31_i32 : i32
  %ub = arith.divsi %n, %c32_i32 : i32
  %res = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %cst) -> (tensor<64xf16>)  : i32 {
    %m0 = arith.muli %iv, %c64_i32 : i32
    %m1 = arith.subi %arg3, %m0 : i32
    %m2 = tt.splat %m1 : i32 -> tensor<64xi32>
    %m = arith.cmpi slt, %r, %m2 : tensor<64xi32>
    %l = tt.load %p, %m, %cst : tensor<64x!tt.ptr<f16>>
    %a = arith.addf %acc, %l : tensor<64xf16>
    scf.yield %a : tensor<64xf16>
  }
  tt.return %res : tensor<64xf16>
}

// CHECK-LABEL: @ub_mask_mismatch
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// CHECK:         tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<64x!tt.ptr<f16>>
// CHECK-NOT:   scf.if
