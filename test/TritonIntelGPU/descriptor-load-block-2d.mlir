// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test: MakeTensorDescOp lowers to LLVM struct packing and
// DescriptorLoadOp with dot_op A encoding generates 2D block loads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_dot_a
  tt.func public @descriptor_load_dot_a(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    // Verify struct packing for MakeTensorDescOp:
    //   struct { i64 shape[2], i64 stride[2], ptr<1> base }
    // CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x32xf16>>
    // Verify 2D block loads are generated for dot A operand.
    // For DPAS A with f16: tile_height=8, tile_width=16, v_blocks=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with dot_op B encoding generates 2D block loads
// with VNNI transform.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_dot_b_vnni
  tt.func public @descriptor_load_dot_b_vnni(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x64xf16>>
    // For DPAS B with f16: tile_height=32, tile_width=16, v_blocks=1, vnni_transform=true.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with f32 element type and dot_op A encoding.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_f32_dot_a
  tt.func public @descriptor_load_f32_dot_a(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f32>, <tensor<64x32xf32>>
    // For DPAS A with f32: tile_height=8, tile_width=8, v_blocks=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] : !tt.tensordesc<tensor<64x32xf32>> -> tensor<64x32xf32, #dot0>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp feeds into tt.dot, verifying the full matmul pattern.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_matmul
  tt.func public @descriptor_load_matmul(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i64, %arg6: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %descA = tt.make_tensor_descriptor %arg0, [%arg2, %arg4], [%arg5, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg4, %arg3], [%arg6, %c1_i64] : <f16>, <tensor<32x64xf16>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = ttg.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}
