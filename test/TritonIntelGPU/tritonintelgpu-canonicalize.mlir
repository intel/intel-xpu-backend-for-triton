// RUN: triton-opt %s -split-input-file -canonicalize -verify-diagnostics | FileCheck %s

tt.func @triton_intel_gpu.extract(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) -> tensor<16x8xf16> {
  // CHECK-LABEL: @triton_intel_gpu.extract
  // CHECK-NEXT: tt.return %arg1 : tensor<16x8xf16>
  %tensor = triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %extract = triton_intel_gpu.extract %tensor[1] : tensor<16x16xf16> -> tensor<16x8xf16>
  tt.return %extract : tensor<16x8xf16>
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) -> tensor<16x16xf16> {
  // CHECK-LABEL: @triton_intel_gpu.extract
  // CHECK-NEXT: tt.return %arg0 : tensor<16x16xf16>
  %extract = triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<16x16xf16>
  tt.return %extract : tensor<16x16xf16>
}
