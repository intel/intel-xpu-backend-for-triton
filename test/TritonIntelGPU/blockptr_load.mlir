// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,LARGE-BLOCK-SIZE-TRANS-B
// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm=one_matrix_per_load_for_bt=1 | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,SMALL-BLOCK-SIZE-TRANS-B

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = ttg.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg8: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %ptrX = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    %X = tt.load %ptrX {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-32: llvm.fadd {{.*}}, {{.*}}
    %0 = arith.addf %D, %X : tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_add_transpose_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg8: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %ptrX = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf32, #dpas>>
    // CHECK-NOT: triton_gen.2Dblockload {{.*}}
    %X = tt.load %ptrX {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<64x64xf32, #dpas>>
    %0 = arith.addf %D, %X : tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #dot1>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-4: triton_gen.dpas {{.*}} {pa = tf32, pb = tf32, rc = 8}
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf32, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x64xf32, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x64xf32, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_a_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[PTR_1:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 16, 1, 1>} {
  tt.func public @dot_op_a_2d_load(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_8]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_10]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_11]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_12]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_13]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_22:.*]] = llvm.urem %[[VAL_20]], %[[VAL_21]] : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.udiv %[[VAL_20]], %[[VAL_21]] : i32
    // CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[VAL_22]], %[[VAL_24]] : i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i32:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ELEM_SIZE_IN_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_38]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.add %[[VAL_39]], %[[VAL_37]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_41]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_40]], %[[OFFSET_0]] : i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           triton_gen.2Dblockload %[[BASE]], %[[HEIGHT]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[offsetX_]], %[[offsetY_]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot0>>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot0>>
    %B = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot1>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_b_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[PTR_1:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 16, 1, 1>} {
  tt.func public @dot_op_b_2d_load(%arg1: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_6]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_7]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_8]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_10]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_11]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_12]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_18:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_17]] : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_17]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.urem %[[VAL_19]], %[[VAL_20]] : i32
    // CHECK:           %[[VAL_22:.*]] = llvm.udiv %[[VAL_19]], %[[VAL_20]] : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[VAL_18]], %[[VAL_23]] : i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i32:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ELEM_SIZE_IN_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_37]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.add %[[VAL_38]], %[[VAL_36]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_39]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_40]], %[[OFFSET_0]] : i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           triton_gen.2Dblockload %[[BASE]], %[[HEIGHT]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[offsetX_]], %[[offsetY_]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot1>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot1>>
    %A = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot0>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // COM: One DPAS operand B per load instruction.
        // SMALL-BLOCK-SIZE-TRANS-B-COUNT-8:    triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
      // COM: Two interleaved DPAS operand B per load instruction. Need to shuffle the loaded value to decompose the VNNI format DPAS operand B.
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_68:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_69:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_71:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_103:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_104:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_106:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_138:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_139:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_141:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_173:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_174:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_176:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      %45 = tt.load %21 {ttig.block_io = "column_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // CHECK:    triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      %45 = tt.load %21 {ttig.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @non_contiguous_load_dot_layout
  // COM: Check mask is not generated when boundary_check is not set.
  // CHECK-NOT: llvm.icmp "slt"
  tt.func public @non_contiguous_load_dot_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #dot_b>>
      // CHECK-COUNT-64: llvm.load {{.*}} -> i16
      %1 = tt.load %0 : !tt.ptr<tensor<64x16xf16, #dot_b>>
      %2 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #dot_a>>
      // CHECK-COUNT-64: llvm.load {{.*}} -> i16
      %3 = tt.load %2 : !tt.ptr<tensor<64x16xf16, #dot_a>>
      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @blocked_layout
  tt.func public @blocked_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %45 = tt.load %21 {ttig.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #blocked>>
      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @boundary_check
  tt.func public @boundary_check(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK-NOT: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %45 = tt.load %21 : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %46 = tt.load %21 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %47 = tt.load %21 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-32: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %48 = tt.load %21 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>
      tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @invalid_bytes_num_per_row
  tt.func public @invalid_bytes_num_per_row(%arg0: !tt.ptr<f32>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf32, #dot_a>>
      // COM: 32 x 4 = 128 bytes, which is >64 bytes
      // CHECK-NOT:    triton_gen.2Dblockload
      %45 = tt.load %21 {ttig.block_io = "row_major"} : !tt.ptr<tensor<64x16xf32, #dot_a>>
      tt.return
  }
}
