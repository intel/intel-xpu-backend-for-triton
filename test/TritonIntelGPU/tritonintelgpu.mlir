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

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>, %ptr : !tt.ptr<tensor<16x8xf16>>) {
  // CHECK-LABEL: @triton_intel_gpu.extract
  // CHECK: %0 = triton_intel_gpu.extract %arg0[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  // CHECK: %1 = triton_intel_gpu.extract %arg1[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  %tensorRes = triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  %ptrRes = triton_intel_gpu.extract %ptr[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  tt.return
}
