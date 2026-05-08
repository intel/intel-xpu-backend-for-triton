// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_GATHER_SCATTER_REWRITE=1 triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s

// Test that scf.for carrying tensor descriptors with candidate MakeTensorDescOps
// is NOT converted to pointer-based operations. The descriptor load/store and the
// loop should remain in tensor descriptor form.

module {
  // CHECK-LABEL: tt.func @for_loop_carried_desc
  tt.func @for_loop_carried_desc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) -> tensor<8x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %stride = arith.extsi %N : i32 to i64
    %desc = tt.make_tensor_descriptor %arg0, [%M, %N], [%stride, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<8x128xf32>
    // CHECK: tt.make_tensor_descriptor
    // CHECK: scf.for
    %result:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc, %acc = %c0_i32) -> (!tt.tensordesc<8x128xf32>, i32) : i32 {
      // CHECK: tt.descriptor_load
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %acc] : !tt.tensordesc<8x128xf32> -> tensor<8x128xf32>
      tt.descriptor_store %iter_desc[%c0_i32, %acc], %ld : !tt.tensordesc<8x128xf32>, tensor<8x128xf32>
      %next = arith.addi %acc, %c1_i32 : i32
      scf.yield %iter_desc, %next : !tt.tensordesc<8x128xf32>, i32
    }
    %final = tt.descriptor_load %result#0[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xf32> -> tensor<8x128xf32>
    tt.return %final : tensor<8x128xf32>
  }
}

// -----

// Test that scf.for with loop-carried descriptor reassigned inside the loop
// (two MakeTensorDescOps) preserves tensor descriptor form.

module {
  // CHECK-LABEL: tt.func @for_loop_carried_reassigned_desc
  tt.func @for_loop_carried_reassigned_desc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<8x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %stride = arith.extsi %N : i32 to i64
    %desc_outer = tt.make_tensor_descriptor %arg0, [%M, %N], [%stride, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<8x128xf32>
    // CHECK: tt.make_tensor_descriptor
    // CHECK: scf.for
    %result:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc_outer, %acc = %c0_i32) -> (!tt.tensordesc<8x128xf32>, i32) : i32 {
      // CHECK: tt.descriptor_load
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %acc] : !tt.tensordesc<8x128xf32> -> tensor<8x128xf32>
      tt.descriptor_store %iter_desc[%c0_i32, %acc], %ld : !tt.tensordesc<8x128xf32>, tensor<8x128xf32>
      // CHECK: tt.make_tensor_descriptor
      %desc_inner = tt.make_tensor_descriptor %arg0, [%M, %N], [%stride, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<8x128xf32>
      %new_desc = arith.select %cond, %desc_inner, %iter_desc : !tt.tensordesc<8x128xf32>
      %next = arith.addi %acc, %c1_i32 : i32
      scf.yield %new_desc, %next : !tt.tensordesc<8x128xf32>, i32
    }
    %final = tt.descriptor_load %result#0[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xf32> -> tensor<8x128xf32>
    tt.return %final : tensor<8x128xf32>
  }
}
