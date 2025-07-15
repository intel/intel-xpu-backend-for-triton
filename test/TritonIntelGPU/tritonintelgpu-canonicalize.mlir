// RUN: triton-opt %s -split-input-file -canonicalize -verify-diagnostics | FileCheck %s

tt.func @ttig.extract(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) -> tensor<16x8xf16> {
  // CHECK-LABEL: @ttig.extract
  // CHECK-NEXT: tt.return %arg1 : tensor<16x8xf16>
  %tensor = ttig.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %extract = ttig.extract %tensor[1] : tensor<16x16xf16> -> tensor<16x8xf16>
  tt.return %extract : tensor<16x8xf16>
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>) -> tensor<16x16xf16> {
  // CHECK-LABEL: @ttig.extract
  // CHECK-NEXT: tt.return %arg0 : tensor<16x16xf16>
  %extract = ttig.extract %tensor[0] : tensor<16x16xf16> -> tensor<16x16xf16>
  tt.return %extract : tensor<16x16xf16>
}
