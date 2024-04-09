// RUN: triton-opt -split-input-file -verify-diagnostics %s

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op requires at least 2 operands}}
  triton_intel_gpu.concat %tensor1 {dim = 0 : i32} : (tensor<16x8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%ptr1 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op requires at least 2 operands}}
  triton_intel_gpu.concat %ptr1 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op rank of concatenated operands must match result rank}}
  triton_intel_gpu.concat %tensor1, %tensor2 {dim = 0 : i32} : (tensor<16x8xf16>, tensor<8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op rank of concatenated operands must match result rank}}
  triton_intel_gpu.concat %ptr1, %ptr2 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op concatenation dim must be less than the tensor rank}}
  triton_intel_gpu.concat %tensor1, %tensor2 {dim = 2 : i32} : (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op concatenation dim must be less than the tensor rank}}  
  triton_intel_gpu.concat %ptr1, %ptr2 {dim = 2 : i32} : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8x8xf32>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op operands and result element type must match}}
  triton_intel_gpu.concat %tensor1, %tensor2 {dim = 0 : i32} : (tensor<16x8xf16>, tensor<8x8xf32>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x16xf32>>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op operands and result element type must match}}  
  triton_intel_gpu.concat %ptr1, %ptr2 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf32>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%tensor1 : tensor<16x16xf16>, %tensor2 : tensor<8x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op static concatenation size mismatch along non-concatenated dimension 1}}
  triton_intel_gpu.concat %tensor1, %tensor2 {dim = 0 : i32} : (tensor<16x16xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.concat(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<8x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.concat' op static concatenation size mismatch along non-concatenated dimension 0}}
  triton_intel_gpu.concat %ptr1, %ptr2 {dim = 1 : i32} : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<8x16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}
