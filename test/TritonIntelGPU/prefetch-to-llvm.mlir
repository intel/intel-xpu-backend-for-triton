// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm --cse -canonicalize | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_block_ptr(
// CHECK-SAME:                                                %[[BASE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                                %[[BASE_HEIGHT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[BASE_WIDTH:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[ROW_STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @prefetch_block_ptr(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // CHECK-DAG:       %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:       %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       %[[CST_2_I32:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:       %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-DAG:       %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:       %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:       %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_15:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_16:.*]] = llvm.zext %[[VAL_15]] : i32 to i64
    // CHECK:           %[[VAL_17:.*]] = llvm.trunc %[[VAL_16]] : i64 to i32
    // CHECK:           %[[VAL_18:.*]] = llvm.urem %[[VAL_17]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.udiv %[[VAL_17]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.urem %[[VAL_19]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.mul %[[BASE_WIDTH]], %[[CST_2]] : i64
    // CHECK:           %[[ROW_MAJOR_BASE_WIDTH:.*]] = llvm.trunc %[[VAL_21]] : i64 to i32
    // CHECK:           %[[ROW_MAJOR_BASE_HEIGHT:.*]] = llvm.trunc %[[BASE_HEIGHT]] : i64 to i32
    // CHECK:           %[[VAL_24:.*]] = llvm.mul %[[ROW_STRIDE]], %[[CST_2]] : i64
    // CHECK:           %[[ROW_MAJOR_PITCH:.*]] = llvm.trunc %[[VAL_24]] : i64 to i32
    // CHECK:           %[[VAL_26:.*]] = llvm.mul %[[VAL_18]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_27:.*]] = llvm.add %[[VAL_26]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_28:.*]] = llvm.urem %[[VAL_27]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.add %[[VAL_28]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_30:.*]] = llvm.mul %[[VAL_20]], %[[CST_2_I32]] : i32
    // CHECK:           %[[VAL_31:.*]] = llvm.add %[[VAL_30]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_32:.*]] = llvm.urem %[[VAL_31]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.add %[[VAL_32]], %[[CST_0]] : i32
    // CHECK:           %[[ROW_MAJOR_OFFSET_Y:.*]] = llvm.trunc %[[VAL_33]] : i32 to i32
    // CHECK:           %[[ROW_MAJOR_OFFSET_X:.*]] = llvm.trunc %[[VAL_29]] : i32 to i32

    // CHECK:           %[[VAL_36:.*]] = llvm.insertelement {{.*}}, {{.*}} : i32] : vector<2xi32>
    // CHECK:           %[[ROW_MAJOR_OFFSETS:.*]] = llvm.insertelement %[[ROW_MAJOR_OFFSET_Y]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[BASE]], %{{.*}}, %[[ROW_MAJOR_BASE_HEIGHT]], %[[ROW_MAJOR_PITCH]], %[[ROW_MAJOR_OFFSETS]])
    %rowMajorPtr = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    ttig.prefetch %rowMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x32xf16>>

    // CHECK:           %[[VAL_32:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_33:.*]] = llvm.zext %[[VAL_32]] : i32 to i64
    // CHECK:           %[[VAL_34:.*]] = llvm.trunc %[[VAL_33]] : i64 to i32
    // CHECK:           %[[VAL_35:.*]] = llvm.urem %[[VAL_34]], %[[CST_2_I32]] : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.udiv %[[VAL_34]], %[[CST_2_I32]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.urem %[[VAL_36]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.mul %[[BASE_WIDTH]], %[[CST_2]] : i64
    // CHECK:           %[[COL_MAJOR_BASE_WIDTH:.*]] = llvm.trunc %[[VAL_38]] : i64 to i32
    // CHECK:           %[[COL_MAJOR_BASE_HEIGHT:.*]] = llvm.trunc %[[BASE_HEIGHT]] : i64 to i32
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[ROW_STRIDE]], %[[CST_2]] : i64
    // CHECK:           %[[COL_MAJOR_PITCH:.*]] = llvm.trunc %[[VAL_41]] : i64 to i32
    // CHECK:           %[[VAL_43:.*]] = llvm.mul %[[VAL_35]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_44:.*]] = llvm.add %[[VAL_43]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_45:.*]] = llvm.urem %[[VAL_44]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_46:.*]] = llvm.add %[[VAL_45]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.mul %[[VAL_37]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_48:.*]] = llvm.add %[[VAL_47]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_49:.*]] = llvm.urem %[[VAL_48]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.add %[[VAL_49]], %[[CST_0]] : i32
    // CHECK:           %[[COL_MAJOR_OFFSET_Y:.*]] = llvm.trunc %[[VAL_50]] : i32 to i32
    // CHECK:           %[[COL_MAJOR_OFFSET_X:.*]] = llvm.trunc %[[VAL_46]] : i32 to i32
    // CHECK:           %[[VAL_54:.*]] = llvm.insertelement %{{.*}}, {{.*}} : i32] : vector<2xi32>
    // CHECK:           %[[COL_MAJOR_OFFSETS:.*]] = llvm.insertelement %[[COL_MAJOR_OFFSET_Y]], {{.*}} : i32] : vector<2xi32>
    // CHECK:           llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[BASE]], %{{.*}}, %[[COL_MAJOR_BASE_HEIGHT]], %[[COL_MAJOR_PITCH]], %[[COL_MAJOR_OFFSETS]]) {{.*}}
    %columnMajorPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%c1_i64, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x16xf16>>
    ttig.prefetch %columnMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "column_major"} : !tt.ptr<tensor<32x16xf16>>

    // COM: The memory is not structured densely. Not to prefetch it to the cache.
    // CHECK-NOT: block_prefetch
    %nonContiguousPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%arg5, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>>
    ttig.prefetch %nonContiguousPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>>
    tt.return
  }
}

// -----

// CHECK: llvm.func spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%tensor_of_ptr: tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>) {
    // CHECK-DAG: %[[MASK:.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK-DAG: %[[VAL_2:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK-DAG: %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[BASE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG: %[[BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1

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
    // CHECK: %[[PITCH_IN_BYTES_I32:.*]] = llvm.intr.umax(%[[UNIFIED_PITCH_I32]], %[[BASE_WIDTH]]) : (i32, i32) -> i32

    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_13]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_27:.*]] = llvm.insertelement {{.*}}, {{.*}} : vector<2xi32>
    // CHECK: %[[OFFSETS:.*]] = llvm.insertelement %[[OFFSET_Y]], {{.*}} : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[VAL_26]], %{{.*}}, %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[OFFSETS]])

    // CHECK: %[[VAL_29:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_30:.*]] = llvm.trunc %[[VAL_29]] : i8 to i1
    // CHECK: %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_32:.*]] = llvm.ptrtoint %[[ADDR_16]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_33:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_32]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_34:.*]] = llvm.inttoptr %[[VAL_33]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_35:.*]] = llvm.insertelement %[[VAL_31]], {{.*}} : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[VAL_34]], %{{.*}}, %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_35]])

    // CHECK: %[[VAL_36:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_37:.*]] = llvm.trunc %[[VAL_36]] : i8 to i1
    // CHECK: %[[VAL_38:.*]] = llvm.select %[[VAL_37]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_39:.*]] = llvm.ptrtoint %[[ADDR_32]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_40:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_39]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_41:.*]] = llvm.inttoptr %[[VAL_40]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_42:.*]] = llvm.insertelement %[[VAL_38]], {{.*}} : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[VAL_41]], %{{.*}}, %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_42]])

    // CHECK: %[[VAL_43:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[MASK]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i8 to i1
    // CHECK: %[[VAL_45:.*]] = llvm.select %[[VAL_44]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[VAL_46:.*]] = llvm.ptrtoint %[[ADDR_48]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_47:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_46]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_48:.*]] = llvm.inttoptr %[[VAL_47]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAL_49:.*]] = llvm.insertelement %[[VAL_45]], {{.*}} : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[VAL_48]], %{{.*}}, %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[VAL_49]])

    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    ttig.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-COUNT-4: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i

    ttig.prefetch %tensor_of_ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    tt.return
  }
}

// -----

// COM: Check that pitch is a constant calculated by AxisInfo analysis, instead of calculating dynamically.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%arg0: i32, %arg1: !tt.ptr<bf16>) {
    %cst_0 = arith.constant dense<512> : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %cst_1 = arith.constant dense<512> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.muli %arg0, %c128_i32 : i32
    %1 = tt.splat %0 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %3 = arith.addi %1, %2 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %4 = arith.remsi %3, %cst_1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = arith.muli %5, %cst_0 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %6 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.addptr %8, %7 : tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-DAG: %[[PITCH:.*]] = llvm.mlir.constant(1024 : i32) : i32
    // CHECK-COUNT-4: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %[[PITCH]], %{{.*}})
    ttig.prefetch %9 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Currently the prefetch operation in this test cannot be lowered correctly, so we check that the test compiles cleanly and not 2D block prefetch operation gets generated.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 8], B = [8, 16], C = [32, 16]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel
  tt.func public @kernel(%arg0 : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>) {
    // CHECK-NOT: __spirv_Subgroup2DBlockPrefetchINTEL
    ttig.prefetch %arg0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    tt.return
  }
}
