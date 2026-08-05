// RUN: triton-opt %s -split-input-file --triton-intel-descriptor-versioning | FileCheck %s

module {
  // CHECK-LABEL: tt.func public @version_runtime_k
  tt.func public @version_runtime_k(%base: !tt.ptr<f16>, %cbase: !tt.ptr<f16>, %gm: i32, %gk_ptr: !tt.ptr<i32>, %gn_ptr: !tt.ptr<i32>, %lda: i64, %ldc: i64, %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[GK:.*]] = tt.load %arg3
    // CHECK-DAG: %[[GN:.*]] = tt.load %arg4
    %gk = tt.load %gk_ptr : !tt.ptr<i32>
    %gn = tt.load %gn_ptr : !tt.ptr<i32>

    // CHECK: %[[ARAW:.*]] = tt.make_tensor_descriptor %arg0, [%arg2, %[[GK]]]
    // CHECK: %[[ADIV:.*]] = arith.divsi %[[GK]], %[[D0:.*]] :
    // CHECK: %[[AK:.*]] = arith.muli %[[ADIV]], %[[D0]] :
    // CHECK: %[[AALIGNED:.*]] = tt.make_tensor_descriptor %arg0, [%arg2, %[[AK]]]
    // CHECK: %[[CRAW:.*]] = tt.make_tensor_descriptor %arg1, [%arg2, %[[GN]]]
    // CHECK: %[[CDIV:.*]] = arith.divsi %[[GN]], %[[D1:.*]] :
    // CHECK: %[[CK:.*]] = arith.muli %[[CDIV]], %[[D1]] :
    // CHECK: %[[CALIGNED:.*]] = tt.make_tensor_descriptor %arg1, [%arg2, %[[CK]]]
    %adesc = tt.make_tensor_descriptor %base, [%gm, %gk], [%lda, %c1_i64] : <f16>, <128x64xf16>
    %cdesc = tt.make_tensor_descriptor %cbase, [%gm, %gn], [%ldc, %c1_i64] : <f16>, <128x128xf16>

    // CHECK: arith.remsi %[[GK]]
    // CHECK: %[[C0:.*]] = arith.cmpi eq
    // CHECK: arith.remsi %[[GN]]
    // CHECK: %[[C1:.*]] = arith.cmpi eq
    // CHECK: %[[PRED:.*]] = arith.andi
    // CHECK: scf.if %[[PRED]]
    // CHECK: scf.while
    // CHECK: scf.for
    // CHECK: tt.descriptor_load %[[AALIGNED]]
    // CHECK: tt.descriptor_store %[[CALIGNED]]
    // CHECK: else
    // CHECK: scf.while
    // CHECK: tt.descriptor_load %[[ARAW]]
    // CHECK: tt.descriptor_store %[[CRAW]]
    %r = scf.while (%t = %c0_i32) : (i32) -> i32 {
      %c = arith.cmpi slt, %t, %ub : i32
      scf.condition(%c) %t : i32
    } do {
    ^bb0(%t: i32):
      %acc = scf.for %kk = %c0_i32 to %ub step %c1_i32 iter_args(%a = %cst) -> (tensor<128x128xf32>) : i32 {
        %off = arith.muli %kk, %c64_i32 : i32
        %ld = tt.descriptor_load %adesc[%c0_i32, %off] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
        scf.yield %a : tensor<128x128xf32>
      }
      %cv = arith.truncf %acc : tensor<128x128xf32> to tensor<128x128xf16>
      tt.descriptor_store %cdesc[%c0_i32, %c0_i32], %cv : !tt.tensordesc<128x128xf16>, tensor<128x128xf16>
      %tn = arith.addi %t, %c64_i32 : i32
      scf.yield %tn : i32
    }
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: tt.func public @constant_k_not_versioned
  tt.func public @constant_k_not_versioned(%base: !tt.ptr<f16>, %gm: i32, %lda: i64, %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    // CHECK-NOT: scf.if
    %adesc = tt.make_tensor_descriptor %base, [%gm, %c64_i32], [%lda, %c1_i64] : <f16>, <128x64xf16>
    %r = scf.while (%t = %c0_i32) : (i32) -> i32 {
      %c = arith.cmpi slt, %t, %ub : i32
      scf.condition(%c) %t : i32
    } do {
    ^bb0(%t: i32):
      %acc = scf.for %kk = %c0_i32 to %ub step %c1_i32 iter_args(%a = %cst) -> (tensor<128x128xf32>) : i32 {
        %off = arith.muli %kk, %c64_i32 : i32
        %ld = tt.descriptor_load %adesc[%c0_i32, %off] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
        scf.yield %a : tensor<128x128xf32>
      }
      %tn = arith.addi %t, %c64_i32 : i32
      scf.yield %tn : i32
    }
    tt.return
  }
}
