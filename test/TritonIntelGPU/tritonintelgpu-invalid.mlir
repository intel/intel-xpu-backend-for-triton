// RUN: triton-opt -split-input-file -verify-diagnostics %s

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op requires at least 2 operands}}
  triton_intel_gpu.glue %tensor1 : (tensor<16x8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op requires at least 2 operands}}
  triton_intel_gpu.glue %ptr1 : (!tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands and result must have the same rank}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands and result must have the same rank}}
  triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf32>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands and result element type must match}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf32>) -> tensor<16x16xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf32>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands and result element type must match}}
  triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf32>>) -> !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<8x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands must have the same shape}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands must have the same shape}}
  triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf16>>) -> !tt.ptr<tensor<16x24xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands cannot exceed result size along any dimension}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<8x16xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands cannot exceed result size along any dimension}}
  triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<8x16xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands cannot be glued along axis 0}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<40x8xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : !tt.ptr<tensor<16x8xf16>>, %tensor2 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands cannot be glued along axis 1}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<32x10xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>, %tensor3 : tensor<16x8xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op glued operands do not exactly cover the result shape}}
  triton_intel_gpu.glue %tensor1, %tensor1, %tensor3 : (tensor<16x8xf16>, tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<32x32xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result must have the same rank}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result must have the same rank}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<4xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result element type must match}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf32>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result element type must match}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x8xf32>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand shape cannot be smaller than result shape along dimension 0}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<32x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand shape cannot be smaller than result shape along dimension 1}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x32xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operands shape is not divisible by result shape along dimension 0}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<12x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operands shape is not divisible by result shape along dimension 1}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x12xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op index must be less than 16}}
  triton_intel_gpu.extract %tensor[16] : tensor<16x16xf16> -> tensor<4x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op index must be less than 2}}
  triton_intel_gpu.extract %ptr[2] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x16xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result must have the same rank}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result must have the same rank}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<4xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result element type must match}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf32>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result element type must match}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x8xf32>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand shape cannot be smaller than result shape along dimension 0}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<32x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand shape cannot be smaller than result shape along dimension 1}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x32xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operands shape is not divisible by result shape along dimension 0}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<12x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operands shape is not divisible by result shape along dimension 1}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x12xf16>>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op index must be less than 16}}
  triton_intel_gpu.extract %tensor[16] : tensor<16x16xf16> -> tensor<4x4xf16>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op index must be less than 2}}
  triton_intel_gpu.extract %ptr[2] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x16xf16>>
  tt.return
}
