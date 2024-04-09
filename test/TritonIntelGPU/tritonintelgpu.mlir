// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8x8xf16>, 
                                 %ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x16xf16>>) {
  // CHECK-LABEL: @triton_intel_gpu.concat
  // CHECK: %0 = triton_intel_gpu.concat %arg0, %arg1 {dim = 0 : i32} : (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
  // CHECK: %1 = triton_intel_gpu.concat %arg2, %arg3 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>) -> !tt.ptr<tensor<16x24xf16>, 1>
  %tensor = triton_intel_gpu.concat %tensor1, %tensor2 {dim = 0 : i32} : (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
  %ptr = triton_intel_gpu.concat %ptr1, %ptr2 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}


