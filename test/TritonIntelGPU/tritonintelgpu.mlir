// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>,
                               %ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf16>>) {
  // CHECK-LABEL: @triton_intel_gpu.glue
  // CHECK: %0 = triton_intel_gpu.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  // CHECK: %1 = triton_intel_gpu.glue %arg2, %arg3 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x16xf16>>
  %tensor = triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %ptr = triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>, %ptr : !tt.ptr<tensor<16x8xf16>>) {
  // CHECK-LABEL: @triton_intel_gpu.extract
  // CHECK: %0 = triton_intel_gpu.extract %arg0[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  // CHECK: %1 = triton_intel_gpu.extract %arg1[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  %tensorRes = triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  %ptrRes = triton_intel_gpu.extract %ptr[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  tt.return
}

// -----

tt.func @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                          %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK:      [[GLUE:%.*]] = triton_intel_gpu.glue
  // CHECK-NEXT: [[RES:%.*]] = scf.for {{.*}} iter_args([[INIT1:%.*]] = [[GLUE]]) -> (tensor<16x16xf16>) : i32 {
  // CHECK:        scf.yield {{.*}} : tensor<16x16xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[RES]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = triton_intel_gpu.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = triton_intel_gpu.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = triton_intel_gpu.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = triton_intel_gpu.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %res {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}
