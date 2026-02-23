// RUN: triton-opt -split-input-file -verify-diagnostics %s
// -----

tt.func @ttig.prefetch(%arg0: !tt.ptr<tensor<2x32xf32>>, %arg1: tensor<4x32xi1>) {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses: 'tensor<2x32xi1>' vs 'tensor<4x32xi1>'}}
  ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  tt.return
}

// -----

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  tt.func @ttig.sub_group_transpose.encoding(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<16x16xf16, #warp>) -> tensor<16x16xf16, #warp> {
    // expected-error @below {{'ttig.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<16x16xf16, #warp>
    tt.return %res : tensor<16x16xf16, #warp>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  tt.func @ttig.sub_group_transpose.shape(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<8x16xf16>) -> tensor<8x16xf16> {
    // expected-error @below {{'ttig.sub_group_transpose' op can only be used on tensors of shape <sub_group_size x sub_group_size> with no encoding}}
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<8x16xf16>
    tt.return %res : tensor<8x16xf16>
  }
}


// -----

// COM: A operand mismatch
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32>,
  %a_mat:tensor<32x16xf32, #dot_operand_a>, %b_mat:tensor<16x32xf32, #dot_operand_b>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>

    // expected-error @+1 {{Layout has opsPerChannel = 2 but tensor element type is 'f32'. Expected 16 bit type.}}
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>

    tt.return
  }
}

// -----

// COM: B operand mismatch
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32>,
  %a_mat:tensor<32x16xf16, #dot_operand_a>, %b_mat:tensor<16x32xf32, #dot_operand_b>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>

    // expected-error @+1 {{Layout has opsPerChannel = 2 but tensor element type is 'f32'. Expected 16 bit type.}}
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf16, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
// expected-error @below {{ttg.dot_op kWidth parameter must match the parent's opsPerChannel}}
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32>,
  %a_mat:tensor<32x16xf32, #dot_operand_a>, %b_mat:tensor<16x32xf32, #dot_operand_b>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 1, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
// expected-error @below {{The DPAS encoding implies an invalid layout for A operand. The non-uniform matrix A could not be referred in kernel}}
#dot_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>

// -----

// expected-error @below {{threadsPerWarp could not be smaller than the execution size}}
#dpas = #ttig.dpas<{repeatCount = 1, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 8, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
