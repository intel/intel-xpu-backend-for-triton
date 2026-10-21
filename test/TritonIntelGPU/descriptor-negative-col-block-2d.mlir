// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: The 2D block message takes x in packed elements. x may be negative, so the
// COM: division by the packing factor must be signed.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.2d_block_io_base_alignment = 4 : i32} {
  // CHECK-LABEL: @packed_load_signed_x
  tt.func public @packed_load_signed_x(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg5: i32) {
    %cm8 = arith.constant -8 : i32
    // CHECK: llvm.sdiv
    // CHECK: triton_gen.2Dblockload {{.*}} transpose = true
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%cm8, %arg5] {column_major} : !tt.ptr<f16> -> tensor<32x32xf16, #dot1>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.2d_block_io_base_alignment = 4 : i32} {
  // CHECK-LABEL: @store_signed_x
  tt.func public @store_signed_x(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %cm8 = arith.constant -8 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    // CHECK: llvm.sdiv
    // CHECK: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %cm8], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// COM: The NaN mask is built in surface space from the logical (pre-compensation)
// COM: coordinates: the width bounds the contiguous dimension (dim 0 for a
// COM: column-major load) and the height the other.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @nan_mask_logical_surface_coordinates(
  // CHECK-SAME: {{.*}}, [[X:%[a-z0-9]+]]: i32, [[Y:%[a-z0-9]+]]: i32
  tt.func public @nan_mask_logical_surface_coordinates(%arg0: !tt.ptr<f16>, %x: i32, %y: i32) {
    %w = arith.constant 128 : i32
    %h = arith.constant 48 : i32
    %p = arith.constant 256 : i32
    // CHECK-DAG: %[[W:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG: %[[H:.*]] = llvm.mlir.constant(48 : i32) : i32
    // CHECK: %[[WELEMS:.*]] = llvm.udiv %[[W]], {{.*}} : i32
    // CHECK: llvm.add {{.*}}, [[X]] : i32
    // CHECK-NEXT: llvm.trunc %[[WELEMS]] : i32 to i32
    // CHECK: llvm.add {{.*}}, [[Y]] : i32
    // CHECK-NEXT: llvm.trunc %[[H]] : i32 to i32
    %0 = ttig.2d_block_load %arg0, %w, %h, %p[%x, %y] {column_major, pad_nan} : !tt.ptr<f16> -> tensor<64x32xf16, #dot1>
    tt.return
  }
}
