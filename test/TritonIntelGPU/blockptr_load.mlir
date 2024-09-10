// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK-COUNT-8: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f({{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = triton_gpu.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=1}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x2cPU3AS1viiiDv2_iPj({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_32b_32r16x1cPU3AS1viiiDv2_iPj({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK-COUNT-4: llvm.call spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_({{.*}}) {{.*}} : (vector<4xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf32, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf32, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x64xf32, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_a_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  tt.func public @dot_op_a_2d_load(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_8:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_8]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_9]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_10]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_11]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_12]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_13]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_14]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_idv()
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.sext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[SUB_GROUP_ID_N:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_17]]  : i32
    // CHECK:           %[[SUB_GROUP_ID_M_:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_17]]  : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[SUB_GROUP_ID_M:.*]] = llvm.urem %[[SUB_GROUP_ID_M_]], %[[VAL_20]]  : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[SUB_GROUP_ID_M]], %[[VAL_23]]  : i32
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
    // CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_33]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.add %[[VAL_34]], %[[VAL_32]] : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_36]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_35]], %[[OFFSET_0]] : i32
    // CHECK:           %[[offsetY:.*]] = llvm.trunc %[[offsetY_]] : i32 to i32
    // CHECK:           %[[offsetX:.*]] = llvm.trunc %[[offsetX_]] : i32 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK:           %[[VAL_48:.*]] = llvm.alloca %[[VAL_47]] x i16 : (i32) -> !llvm.ptr
    // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_51:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK:           %[[VAL_52:.*]] = llvm.insertelement %[[offsetX]], %[[VAL_51]]{{\[}}%[[VAL_50]] : i32] : vector<2xi32>
    // CHECK:           %[[VAL_53:.*]] = llvm.insertelement %[[offsetY]], %[[VAL_52]]{{\[}}%[[VAL_49]] : i32] : vector<2xi32>
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot0>>
    // CHECK:           llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(%[[BASE]], %[[HEIGHT]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[VAL_53]], %[[VAL_48]])
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot0>>
    %B = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot1>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK-DAG:         llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_b_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  tt.func public @dot_op_b_2d_load(%arg1: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_8]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_9]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_10]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_11]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_12]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_13]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_idv()
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.sext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_17:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_16]]  : i32
    // CHECK:           %[[VAL_18:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_16]]  : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.urem %[[VAL_18]], %[[VAL_19]]  : i32
    // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[VAL_17]], %[[VAL_22]]  : i32
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
    // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_32]] : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.add %[[VAL_33]], %[[VAL_31]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_34]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_35]], %[[OFFSET_0]] : i32
    // CHECK:           %[[VAL_42:.*]] = llvm.trunc %[[offsetY_]] : i32 to i32
    // CHECK:           %[[VAL_43:.*]] = llvm.trunc %[[offsetX_]] : i32 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.alloca %[[VAL_46]] x i32 : (i32) -> !llvm.ptr
    // CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK:           %[[VAL_51:.*]] = llvm.insertelement %[[VAL_43]], %[[VAL_50]]{{\[}}%[[VAL_49]] : i32] : vector<2xi32>
    // CHECK:           %[[VAL_52:.*]] = llvm.insertelement %[[VAL_42]], %[[VAL_51]]{{\[}}%[[VAL_48]] : i32] : vector<2xi32>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot1>>
    // CHECK:           llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(%[[BASE]], %[[HEIGHT]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[VAL_52]], %[[VAL_47]])
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot1>>
    %A = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot0>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK:   llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i32
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // CHECK:    llvm.ptrtoint
      // CHECK:    %[[ELEM_BITS:.*]] = llvm.mlir.constant(32 : i32) : i32
      // CHECK:    %[[TILE_WIDTH:.*]] = llvm.mlir.constant(8 : i32) : i32
      // CHECK:    %[[TILE_HEIGHT:.*]] = llvm.mlir.constant(32 : i32) : i32
      // CHECK:    %[[VBLOCKS:.*]] = llvm.mlir.constant(1 : i32) : i32
      // CHECK:    %[[TRANSPOSE:.*]] = llvm.mlir.constant(true) : i1
      // CHECK:    %[[VNNI:.*]] = llvm.mlir.constant(false) : i1
      // CHECK:    %[[VAL_68:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i32({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[ELEM_BITS]], %[[TILE_WIDTH]], %[[TILE_HEIGHT]], %[[VBLOCKS]], %[[TRANSPOSE]], %[[VNNI]], {{.*}})
      // CHECK:    %[[VAL_69:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
      // CHECK:    %[[VAL_71:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      // CHECK:    %[[VAL_103:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i32
      // CHECK:    %[[VAL_104:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
      // CHECK:    %[[VAL_106:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      // CHECK:    %[[VAL_138:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i32
      // CHECK:    %[[VAL_139:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
      // CHECK:    %[[VAL_141:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      // CHECK:    %[[VAL_173:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i32
      // CHECK:    %[[VAL_174:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
      // CHECK:    %[[VAL_176:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

// CHECK:   llvm.func spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // CHECK:    llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @non_contiguous_load_dot_layout
  tt.func public @non_contiguous_load_dot_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
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

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @blocked_layout
  tt.func public @blocked_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #blocked>>
      tt.return
  }
}
