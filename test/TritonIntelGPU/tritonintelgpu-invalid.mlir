// RUN: triton-opt -split-input-file -verify-diagnostics %s

// COM: Ensure that tensors with different shape cannot be glued.
tt.func @triton_intel_gpu.glue(%tensor1 : tensor<16xf16>, %tensor2 : tensor<16x8xf32>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands must have the same type}}
  triton_intel_gpu.glue %tensor1, %tensor2 : (tensor<16xf16>, tensor<16x8xf32>) -> tensor<16x16xf16>
  tt.return
}

// -----

// COM: Ensure that tensors with the different element types cannot be glued.
tt.func @triton_intel_gpu.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf32>>) {
  // expected-error @+1 {{'triton_intel_gpu.glue' op operands must have the same type}}
  triton_intel_gpu.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf32>>) -> !tt.ptr<tensor<16x16xf16>>
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
  // expected-error @+1 {{'triton_intel_gpu.extract' op operand and result element type must match}}
  triton_intel_gpu.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf32>
  tt.return
}

// -----

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<16xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op result rank cannot be greater than operand rank}}
  triton_intel_gpu.extract %ptr[0] : !tt.ptr<tensor<16xf16>> -> !tt.ptr<tensor<2x8xf16>>
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

tt.func @triton_intel_gpu.extract(%ptr : !tt.ptr<tensor<32x32xf16>>) {
  // expected-error @+1 {{'triton_intel_gpu.extract' op operands shape is not divisible by result shape along dimension 1}}
  triton_intel_gpu.extract %ptr[2] : !tt.ptr<tensor<32x32xf16>> -> !tt.ptr<tensor<24xf16>>
  tt.return
}

// -----

#warp = #triton_intel_gpu.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  tt.func @triton_intel_gpu.sub_group_transpose.encoding(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<16x16xf16, #warp>) -> tensor<16x16xf16, #warp> {
    // expected-error @below {{'triton_intel_gpu.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = triton_intel_gpu.sub_group_transpose %local_buffer, %src : tensor<16x16xf16, #warp>
    tt.return %res : tensor<16x16xf16, #warp>
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  tt.func @triton_intel_gpu.sub_group_transpose.shape(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<8x16xf16>) -> tensor<8x16xf16> {
    // expected-error @below {{'triton_intel_gpu.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = triton_intel_gpu.sub_group_transpose %local_buffer, %src : tensor<8x16xf16>
    tt.return %res : tensor<8x16xf16>
  }
}
