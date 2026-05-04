// RUN: triton-opt %s -split-input-file -triton-licm | FileCheck %s

// COM: Case A: An op carrying the {tt.no_licm} unit attribute must NOT be hoisted
// COM: out of the loop even when its operands are loop-invariant. The arith.addi
// COM: stays inside the scf.for body.

// CHECK-LABEL: @no_licm_attr_blocks_hoist
tt.func @no_licm_attr_blocks_hoist(%arg0: i32, %arg1: i32, %lb: i32, %ub: i32, %step: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: scf.for
  // CHECK:   arith.addi %{{.*}}, %{{.*}} {tt.no_licm} : i32
  // CHECK:   scf.yield
  %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c0_i32) -> (i32) : i32 {
    %sum = arith.addi %arg0, %arg1 {tt.no_licm} : i32
    %new_acc = arith.addi %acc, %sum : i32
    scf.yield %new_acc : i32
  }
  tt.return %result : i32
}

// -----

// COM: Case B: The same loop-invariant arith.addi without the {tt.no_licm}
// COM: attribute is hoisted out of the loop by triton-licm. It appears before
// COM: scf.for in the output.

// CHECK-LABEL: @no_attr_gets_hoisted
tt.func @no_attr_gets_hoisted(%arg0: i32, %arg1: i32, %lb: i32, %ub: i32, %step: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: arith.addi %{{.*}}, %{{.*}} : i32
  // CHECK: scf.for
  // CHECK-NOT: arith.addi %arg0, %arg1
  %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c0_i32) -> (i32) : i32 {
    %sum = arith.addi %arg0, %arg1 : i32
    %new_acc = arith.addi %acc, %sum : i32
    scf.yield %new_acc : i32
  }
  tt.return %result : i32
}

// -----

// COM: Case C: A loop-invariant tt.load tagged with {tt.no_licm} must NOT be
// COM: hoisted either. The attribute check in shouldMoveOutOfRegion runs before
// COM: the LoadOp-specific legality checks, so the load stays inside scf.for.

// CHECK-LABEL: @no_licm_attr_blocks_load_hoist
tt.func @no_licm_attr_blocks_load_hoist(%arg0: tensor<1024x!tt.ptr<f32>>, %lb: i32, %ub: i32, %step: i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  // CHECK: scf.for
  // CHECK:   tt.load %{{.*}} {tt.no_licm}
  // CHECK:   scf.yield
  %1 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<1024xf32>) : i32 {
    %val = tt.load %arg0 {tt.no_licm} : tensor<1024x!tt.ptr<f32>>
    %sum = arith.addf %acc, %val : tensor<1024xf32>
    scf.yield %sum : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}
