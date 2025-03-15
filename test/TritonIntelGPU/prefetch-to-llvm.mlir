// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s

// CHECK-DAG: llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_2r16x2cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_block_ptr(
// CHECK-SAME:                                                %[[BASE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                                %[[BASE_HEIGHT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[BASE_WIDTH:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[ROW_STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @prefetch_block_ptr(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_2_I32:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_15:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_16:.*]] = llvm.zext %[[VAL_15]] : i32 to i64
    // CHECK:           %[[VAL_17:.*]] = llvm.trunc %[[VAL_16]] : i64 to i32
    // CHECK:           %[[VAL_18:.*]] = llvm.urem %[[VAL_17]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.udiv %[[VAL_17]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.urem %[[VAL_19]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.mul %[[BASE_WIDTH]], %[[CST_2]] : i64
    // CHECK:           %[[ROW_MAJOR_BASE_WIDTH_I32:.*]] = llvm.trunc %[[VAL_21]] : i64 to i32
    // CHECK:           %[[ROW_MAJOR_BASE_HEIGHT_I32:.*]] = llvm.trunc %[[BASE_HEIGHT]] : i64 to i32
    // CHECK:           %[[VAL_24:.*]] = llvm.mul %[[ROW_STRIDE]], %[[CST_2]] : i64
    // CHECK:           %[[PITCH:.*]] = llvm.trunc %[[VAL_24]] : i64 to i32
    // CHECK:           %[[VAL_26:.*]] = llvm.mul %[[VAL_18]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_27:.*]] = llvm.add %[[VAL_26]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_28:.*]] = llvm.urem %[[VAL_27]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.add %[[VAL_28]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_30:.*]] = llvm.mul %[[VAL_20]], %[[CST_2_I32]] : i32
    // CHECK:           %[[VAL_31:.*]] = llvm.add %[[VAL_30]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_32:.*]] = llvm.urem %[[VAL_31]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.add %[[VAL_32]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.trunc %[[VAL_33]] : i32 to i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.trunc %[[VAL_29]] : i32 to i32
    // CHECK:           %[[VAL_36:.*]] = llvm.insertelement %[[OFFSET_X]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           %[[OFFSETS:.*]] = llvm.insertelement %[[OFFSET_Y]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_2r16x2cPU3AS1viiiDv2_i(%[[BASE]], %[[ROW_MAJOR_BASE_WIDTH_I32]], %[[ROW_MAJOR_BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSETS]]) {{.*}}
    %rowMajorPtr = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    triton_intel_gpu.prefetch %rowMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<16x32xf16>>

    // COM: The memory layout is same for the column major memory and row major memory. The prefetch should be the same.

    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.trunc %[[VAL_39]] : i64 to i32
    // CHECK:           %[[VAL_41:.*]] = llvm.urem %[[VAL_40]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_42:.*]] = llvm.udiv %[[VAL_40]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.urem %[[VAL_42]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_44:.*]] = llvm.mul %[[BASE_WIDTH]], %[[CST_2]] : i64
    // CHECK:           %[[COLUM_MAJOR_BASE_WIDTH_I32:.*]] = llvm.trunc %[[VAL_44]] : i64 to i32
    // CHECK:           %[[COLUM_MAJOR_BASE_HEIGHT_I32:.*]] = llvm.trunc %[[BASE_HEIGHT]] : i64 to i32
    // CHECK:           %[[VAL_47:.*]] = llvm.mul %[[ROW_STRIDE]], %[[CST_2]] : i64
    // CHECK:           %[[COLUM_MAJOR_PITCH:.*]] = llvm.trunc %[[VAL_47]] : i64 to i32
    // CHECK:           %[[VAL_49:.*]] = llvm.mul %[[VAL_41]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.add %[[VAL_49]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_51:.*]] = llvm.urem %[[VAL_50]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_52:.*]] = llvm.add %[[VAL_51]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_53:.*]] = llvm.mul %[[VAL_43]], %[[CST_2_I32]] : i32
    // CHECK:           %[[VAL_54:.*]] = llvm.add %[[VAL_53]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_55:.*]] = llvm.urem %[[VAL_54]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_56:.*]] = llvm.add %[[VAL_55]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_57:.*]] = llvm.trunc %[[VAL_56]] : i32 to i32
    // CHECK:           %[[VAL_58:.*]] = llvm.trunc %[[VAL_52]] : i32 to i32
    // CHECK:           %[[VAL_59:.*]] = llvm.insertelement %[[VAL_58]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           %[[COLUM_MAJOR_OFFSETS:.*]] = llvm.insertelement %[[VAL_57]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_2r16x2cPU3AS1viiiDv2_i(%[[BASE]], %[[COLUM_MAJOR_BASE_WIDTH_I32]], %[[COLUM_MAJOR_BASE_HEIGHT_I32]], %[[COLUM_MAJOR_PITCH]], %[[COLUM_MAJOR_OFFSETS]]) {{.*}}
    %columnMajorPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%c1_i64, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x16xf16>>
    triton_intel_gpu.prefetch %columnMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<32x16xf16>>

    // COM: The memory is not structured densely. Not to prefetch it to the cache.
    // CHECK-NOT: block_prefetch
    %nonContiguousPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%arg5, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>>
    triton_intel_gpu.prefetch %nonContiguousPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>>
    tt.return
  }
}

// -----

// CHECK:   llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%tensor_of_ptr: tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>) {
    // CHECK: %[[MASK:.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK: %[[VAL_2:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[BASE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: %[[BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1

    // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_16:.*]] = llvm.extractvalue {{.*}}[16] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_32:.*]] = llvm.extractvalue {{.*}}[32] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_48:.*]] = llvm.extractvalue {{.*}}[48] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[VAL_13:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_14:.*]] = llvm.ptrtoint %[[ADDR_1]] : !llvm.ptr<1> to i64
    // CHECK: %[[PITCH:.*]] = llvm.sub %[[VAL_14]], %[[VAL_13]] : i64
    // CHECK: %[[UNIFIED_PITCH:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[PITCH]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFIED_PITCH_I32:.*]] = llvm.trunc %[[UNIFIED_PITCH]] : i64 to i32
    // CHECK: %[[VAL_18:.*]] = llvm.intr.umax(%[[UNIFIED_PITCH_I32]], %[[BASE_WIDTH]]) : (i32, i32) -> i32
    // CHECK: %[[PITCH_IN_BYTES_I32:.*]] = llvm.trunc %[[VAL_18]] : i32 to i32

    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_13]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_27:.*]] = llvm.insertelement %[[CST_0]], {{.*}} : vector<2xi32>
    // CHECK: %[[OFFSETS:.*]] = llvm.insertelement %[[OFFSET_Y]], {{.*}} : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_26]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[OFFSETS]])

    // CHECK: %[[VAL_29:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_30:.*]] = llvm.trunc %[[VAL_29]] : i8 to i1
    // CHECK: %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_32:.*]] = llvm.ptrtoint %[[ADDR_16]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_33:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_32]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_34:.*]] = llvm.inttoptr %[[VAL_33]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_35:.*]] = llvm.insertelement %[[VAL_31]], {{.*}} : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_34]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_35]])

    // CHECK: %[[VAL_36:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_37:.*]] = llvm.trunc %[[VAL_36]] : i8 to i1
    // CHECK: %[[VAL_38:.*]] = llvm.select %[[VAL_37]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_39:.*]] = llvm.ptrtoint %[[ADDR_32]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_40:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_39]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_41:.*]] = llvm.inttoptr %[[VAL_40]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_42:.*]] = llvm.insertelement %[[VAL_38]], {{.*}} : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_41]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_42]])

    // CHECK: %[[VAL_43:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i8 to i1
    // CHECK: %[[VAL_45:.*]] = llvm.select %[[VAL_44]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_46:.*]] = llvm.ptrtoint %[[ADDR_48]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_47:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_46]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_48:.*]] = llvm.inttoptr %[[VAL_47]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_49:.*]] = llvm.insertelement %[[VAL_45]], {{.*}} : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_48]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_49]])

    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    triton_intel_gpu.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK: %[[VAL_52:.*]] = llvm.select %[[TRUE]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_53:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_13]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_54:.*]] = llvm.inttoptr %[[VAL_53]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_55:.*]] = llvm.insertelement %[[VAL_52]], {{.*}} : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_54]], {{.*}}, {{.*}}, {{.*}}, %[[VAL_55]])

    // CHECK: %[[VAL_56:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_32]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_57:.*]] = llvm.inttoptr %[[VAL_56]] : i64 to !llvm.ptr<1>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_57]], {{.*}}, {{.*}}, {{.*}}, %[[VAL_55]])

    // CHECK: %[[VAL_58:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_39]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_59:.*]] = llvm.inttoptr %[[VAL_58]] : i64 to !llvm.ptr<1>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_59]], {{.*}}, {{.*}}, {{.*}}, %[[VAL_55]])

    // CHECK: %[[VAL_60:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_46]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_61:.*]] = llvm.inttoptr %[[VAL_60]] : i64 to !llvm.ptr<1>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%[[VAL_61]], {{.*}}, {{.*}}, {{.*}}, %[[VAL_55]])

    %mask_scalar = arith.constant 1 : i1
    triton_intel_gpu.prefetch %tensor_of_ptr, %mask_scalar {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, i1

    // CHECK-COUNT-4: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i

    triton_intel_gpu.prefetch %tensor_of_ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, triton_intel_gpu.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    tt.return
  }
}
