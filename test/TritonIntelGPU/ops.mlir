// RUN: triton-opt %s | FileCheck %s

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.support_sg_2d_block"} {
  tt.func public @prefetch_op(%tensor_of_ptr: tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>,
                              %block_ptr: !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, 1>) {
    // COM: Mask of tensor type
    // CHECK: triton_intel_gpu.prefetch
    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    triton_intel_gpu.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // COM: Mask of scalar type
    // CHECK: triton_intel_gpu.prefetch
    %mask_scalar = arith.constant 1 : i1
    triton_intel_gpu.prefetch %tensor_of_ptr, %mask_scalar {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, i1

    // COM: Block pointer includes the boundary information. No mask
    // CHECK: triton_intel_gpu.prefetch
    triton_intel_gpu.prefetch %block_ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, 1>
    tt.return
  }
}
