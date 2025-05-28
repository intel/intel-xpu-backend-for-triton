// RUN: triton-opt -split-input-file -verify-diagnostics %s

// COM: Ensure that tensors with different shape cannot be glued.
tt.func @ttig.glue(%tensor1 : tensor<16xf16>, %tensor2 : tensor<16x8xf32>) {
  // expected-error @+1 {{'ttig.glue' op operands must have the same type}}
  ttig.glue %tensor1, %tensor2 : (tensor<16xf16>, tensor<16x8xf32>) -> tensor<16x16xf16>
  tt.return
}

// -----

// COM: Ensure that tensors with the different element types cannot be glued.
tt.func @ttig.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf32>>) {
  // expected-error @+1 {{'ttig.glue' op operands must have the same type}}
  ttig.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf32>>) -> !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

tt.func @ttig.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) {
  // expected-error @+1 {{'ttig.glue' op operands cannot exceed result size along any dimension}}
  ttig.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<8x16xf16>
  tt.return
}

// -----

tt.func @ttig.glue(%ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'ttig.glue' op operands cannot exceed result size along any dimension}}
  ttig.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<8x16xf16>>
  tt.return
}

// -----

tt.func @ttig.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>) {
  // expected-error @+1 {{'ttig.glue' op operands cannot be glued along axis 0}}
  ttig.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<40x8xf16>
  tt.return
}

// -----

tt.func @ttig.glue(%tensor1 : !tt.ptr<tensor<16x8xf16>>, %tensor2 : !tt.ptr<tensor<16x8xf16>>) {
  // expected-error @+1 {{'ttig.glue' op operands cannot be glued along axis 1}}
  ttig.glue %tensor1, %tensor2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<32x10xf16>>
  tt.return
}

// -----

tt.func @ttig.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>, %tensor3 : tensor<16x8xf16>) {
  // expected-error @+1 {{'ttig.glue' op glued operands do not exactly cover the result shape}}
  ttig.glue %tensor1, %tensor1, %tensor3 : (tensor<16x8xf16>, tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<32x32xf16>
  tt.return
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'ttig.extract' op operand and result element type must match}}
  ttig.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf32>
  tt.return
}

// -----

tt.func @ttig.extract(%ptr : !tt.ptr<tensor<16xf16>>) {
  // expected-error @+1 {{'ttig.extract' op result rank cannot be greater than operand rank}}
  ttig.extract %ptr[0] : !tt.ptr<tensor<16xf16>> -> !tt.ptr<tensor<2x8xf16>>
  tt.return
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'ttig.extract' op operand shape cannot be smaller than result shape along dimension 0}}
  ttig.extract %tensor[0] : tensor<16x16xf16> -> tensor<32x4xf16>
  tt.return
}

// -----

tt.func @ttig.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'ttig.extract' op operand shape cannot be smaller than result shape along dimension 1}}
  ttig.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x32xf16>>
  tt.return
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'ttig.extract' op operands shape is not divisible by result shape along dimension 0}}
  ttig.extract %tensor[0] : tensor<16x16xf16> -> tensor<12x4xf16>
  tt.return
}

// -----

tt.func @ttig.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'ttig.extract' op operands shape is not divisible by result shape along dimension 1}}
  ttig.extract %ptr[0] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x12xf16>>
  tt.return
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>) {
  // expected-error @+1 {{'ttig.extract' op index must be less than 16}}
  ttig.extract %tensor[16] : tensor<16x16xf16> -> tensor<4x4xf16>
  tt.return
}

// -----

tt.func @ttig.extract(%ptr : !tt.ptr<tensor<16x16xf16>>) {
  // expected-error @+1 {{'ttig.extract' op index must be less than 2}}
  ttig.extract %ptr[2] : !tt.ptr<tensor<16x16xf16>> -> !tt.ptr<tensor<8x16xf16>>
  tt.return
}

// -----

tt.func @ttig.extract(%ptr : !tt.ptr<tensor<32x32xf16>>) {
  // expected-error @+1 {{'ttig.extract' op operands shape is not divisible by result shape along dimension 1}}
  ttig.extract %ptr[2] : !tt.ptr<tensor<32x32xf16>> -> !tt.ptr<tensor<24xf16>>
  tt.return
}

// -----

tt.func @ttig.prefetch(%arg0: !tt.ptr<tensor<2x32xf32>>, %arg1: tensor<4x32xi1>) {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses: 'tensor<2x32xi1>' vs 'tensor<4x32xi1>'}}
  ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  tt.return
}

// -----

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func @ttig.sub_group_transpose.encoding(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<16x16xf16, #warp>) -> tensor<16x16xf16, #warp> {
    // expected-error @below {{'ttig.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<16x16xf16, #warp>
    tt.return %res : tensor<16x16xf16, #warp>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func @ttig.sub_group_transpose.shape(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<8x16xf16>) -> tensor<8x16xf16> {
    // expected-error @below {{'ttig.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<8x16xf16>
    tt.return %res : tensor<8x16xf16>
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!ttg.memdesc<32x16xf32, #shared, #smem>, %b:!ttg.memdesc<16x32xf32, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xf32, #shared, #smem> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #dot_operand_b>

    // expected-error @+1 {{Layout has opsPerChannel = 2 but tensor element type is 'f32'. Expected 16 bit type.}}
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %38 = ttg.convert_layout %28 : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
// expected-error @below {{ttg.dot_op kWidth parameter must match the parent's opsPerChannel}}
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!ttg.memdesc<32x16xf32, #shared, #smem>, %b:!ttg.memdesc<16x32xf32, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xf32, #shared, #smem> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %38 = ttg.convert_layout %28 : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>

    tt.return
  }
}
