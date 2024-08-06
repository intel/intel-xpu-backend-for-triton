// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm -canonicalize | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_0:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>,
  // CHECK-SAME:    %[[SCRATCH_SLM:.*]]: !llvm.ptr<3>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], {{.*}}} {
  tt.func @convert_dot(%A: tensor<128x64xf16, #blocked0>) {
    // CHECK-DAG:     %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %AA = triton_gpu.local_alloc %A : (tensor<128x64xf16, #blocked0>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of triton_gpu.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK:         %[[LINEAR_WARP_ID:.*]] = llvm.udiv %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[LANE_ID:.*]] = llvm.urem %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[VAL_471:.*]] = llvm.udiv %[[LINEAR_WARP_ID]], %[[CST_8]]  : i32
    // CHECK:         %[[WARP_ID_M:.*]] = llvm.urem %[[VAL_471]], %[[CST_4]]  : i32
    // CHECK:         %[[OUTER_WARP_ID:.*]] = llvm.urem %[[WARP_ID_M]], %[[CST_8]]  : i32
    // COM:   Compute the offsets of the elements on the SLM.
    // CHECK:         %[[repKDimStride:.*]] = llvm.mul %[[CST_16]], %[[CST_1]] : i32
    // CHECK:         %[[repNonKDimStride:.*]] = llvm.mul %[[CST_64]], %[[CST_64]] : i32
    // CHECK:         %[[warpMatStride:.*]] = llvm.mul %[[CST_16]], %[[CST_64]] : i32
    // CHECK:         %[[laneRowIndex:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[laneColIndex_:.*]] = llvm.urem %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[laneColIndex:.*]] = llvm.mul %[[laneColIndex_]], %[[CST_1]] : i32
    // CHECK:         %[[iOff:.*]] = llvm.mul %[[OUTER_WARP_ID]], %[[warpMatStride]] : i32
    // CHECK:         %[[rowIndex:.*]] = llvm.mul %[[CST_0]], %[[CST_1]] : i32
    // CHECK:         %[[iBase_0:.*]] = llvm.add %[[rowIndex]], %[[laneRowIndex]] : i32
    // CHECK:         %[[iBase_1:.*]] = llvm.add %[[iBase_0]], %[[CST_0]] : i32
    // CHECK:         %[[jBase_0:.*]] = llvm.add %[[laneColIndex]], %[[CST_0]] : i32
    // CHECK:         %[[iBase_Rounded:.*]] = llvm.urem %[[iBase_1]], %[[CST_128]]  : i32
    // CHECK:         %[[jBase_Rounded:.*]] = llvm.urem %[[jBase_0]], %[[CST_64]]  : i32
    // CHECK:         %[[VAL_487:.*]] = llvm.udiv %[[iBase_Rounded]], %[[CST_1]]  : i32
    // CHECK:         %[[phase:.*]] = llvm.urem %[[VAL_487]], %[[CST_1]]  : i32
    // COM: swizzle: col_swizzled = (col / vec) ^ phase * vec
    // CHECK:         %[[VAL_489:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:         %[[VAL_490:.*]] = llvm.add %[[VAL_489]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_491:.*]] = llvm.xor %[[VAL_490]], %[[phase]]  : i32
    // CHECK:         %[[jOff:.*]] = llvm.mul %[[VAL_491]], %[[CST_1]] : i32
    // CHECK:         %[[VAL_493:.*]] = llvm.mul %[[iBase_Rounded]], %[[CST_64]] : i32
    // CHECK:         %[[OFFSET_I:.*]] = llvm.add %[[VAL_493]], %[[iOff]] : i32
    // CHECK:         %[[VAL_495:.*]] = llvm.mul %[[jBase_Rounded]], %[[CST_1]] : i32
    // CHECK:         %[[OFFSET_J:.*]] = llvm.add %[[VAL_495]], %[[jOff]] : i32
    // CHECK:         %[[OFFSET:.*]] = llvm.add %[[OFFSET_I]], %[[OFFSET_J]] : i32
    // CHECK:         %[[SLM_SWIZZLE_OFFSET:.*]] = llvm.sub %[[CST_0]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1026:.*]] = llvm.getelementptr %[[SCRATCH_SLM]]{{\[}}%[[SLM_SWIZZLE_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16

    // COM: Total 16 ptrs per repetition cluster [2, 1] for operand A.
    // CHECK:         %[[VAL_1027:.*]] = llvm.getelementptr %[[VAL_1026]]{{\[}}%[[OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-COUNT-15:{{.*}} = llvm.getelementptr %[[VAL_1026]]{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         %[[offsetOuter:.*]] = llvm.mul %[[repNonKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[offsetInner:.*]] = llvm.mul %[[repKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1063:.*]] = llvm.add %[[offsetOuter]], %[[offsetInner]] : i32

    // COM: Total 16 scalar per repetition cluster [2, 1] for operand A.
    // CHECK:         %[[VAL_1064:.*]] = llvm.getelementptr %[[VAL_1027]]{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         llvm.load %[[VAL_1064]] : !llvm.ptr<3> -> f16
    // CHECK-COUNT-15:{{.*}} = llvm.getelementptr {{.*}}{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16{{[[:space:]].*}}{{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> f16

    %AA_DOT = triton_gpu.local_load %AA : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #dot_operand_a>

    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #dot_operand_b>
    %D = tt.dot %AA_DOT, %cst1, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_0:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>,
  // CHECK-SAME:    %[[SCRATCH_SLM:.*]]: !llvm.ptr<3>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], {{.*}}} {
  tt.func @convert_dot(%A: tensor<128x64xf16, #blocked0>) {
    // CHECK-DAG:     %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %AA = triton_gpu.local_alloc %A : (tensor<128x64xf16, #blocked0>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of triton_gpu.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK:         %[[LINEAR_WARP_ID:.*]] = llvm.udiv %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[LANE_ID:.*]] = llvm.urem %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[VAL_471:.*]] = llvm.udiv %[[LINEAR_WARP_ID]], %[[CST_8]]  : i32
    // CHECK:         %[[WARP_ID_M:.*]] = llvm.urem %[[VAL_471]], %[[CST_4]]  : i32
    // CHECK:         %[[OUTER_WARP_ID:.*]] = llvm.urem %[[WARP_ID_M]], %[[CST_4]]  : i32
    // COM:   Compute the offsets of the elements on the SLM.
    // CHECK:         %[[repKDimStride:.*]] = llvm.mul %[[CST_16]], %[[CST_1]] : i32
    // CHECK:         %[[repNonKDimStride:.*]] = llvm.mul %[[CST_128]], %[[CST_64]] : i32
    // CHECK:         %[[warpMatStride:.*]] = llvm.mul %[[CST_32]], %[[CST_64]] : i32
    // CHECK:         %[[laneRowIndex:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[laneColIndex_:.*]] = llvm.urem %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[laneColIndex:.*]] = llvm.mul %[[laneColIndex_]], %[[CST_1]] : i32
    // CHECK:         %[[iOff:.*]] = llvm.mul %[[OUTER_WARP_ID]], %[[warpMatStride]] : i32
    // CHECK:         %[[rowIndex:.*]] = llvm.mul %[[CST_0]], %[[CST_1]] : i32
    // CHECK:         %[[iBase_0:.*]] = llvm.add %[[rowIndex]], %[[laneRowIndex]] : i32
    // CHECK:         %[[iBase_1:.*]] = llvm.add %[[iBase_0]], %[[CST_0]] : i32
    // CHECK:         %[[jBase_0:.*]] = llvm.add %[[laneColIndex]], %[[CST_0]] : i32
    // CHECK:         %[[iBase_Rounded:.*]] = llvm.urem %[[iBase_1]], %[[CST_128]]  : i32
    // CHECK:         %[[jBase_Rounded:.*]] = llvm.urem %[[jBase_0]], %[[CST_64]]  : i32
    // CHECK:         %[[VAL_487:.*]] = llvm.udiv %[[iBase_Rounded]], %[[CST_1]]  : i32
    // CHECK:         %[[phase:.*]] = llvm.urem %[[VAL_487]], %[[CST_1]]  : i32
    // COM: swizzle: col_swizzled = (col / vec) ^ phase * vec
    // CHECK:         %[[VAL_489:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:         %[[VAL_490:.*]] = llvm.add %[[VAL_489]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_491:.*]] = llvm.xor %[[VAL_490]], %[[phase]]  : i32
    // CHECK:         %[[jOff:.*]] = llvm.mul %[[VAL_491]], %[[CST_1]] : i32
    // CHECK:         %[[VAL_493:.*]] = llvm.mul %[[iBase_Rounded]], %[[CST_64]] : i32
    // CHECK:         %[[OFFSET_I:.*]] = llvm.add %[[VAL_493]], %[[iOff]] : i32
    // CHECK:         %[[VAL_495:.*]] = llvm.mul %[[jBase_Rounded]], %[[CST_1]] : i32
    // CHECK:         %[[OFFSET_J:.*]] = llvm.add %[[VAL_495]], %[[jOff]] : i32
    // CHECK:         %[[OFFSET:.*]] = llvm.add %[[OFFSET_I]], %[[OFFSET_J]] : i32
    // CHECK:         %[[SLM_SWIZZLE_OFFSET:.*]] = llvm.sub %[[CST_0]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1026:.*]] = llvm.getelementptr %[[SCRATCH_SLM]]{{\[}}%[[SLM_SWIZZLE_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16

    // COM: Total 32 ptrs per repetition cluster [4, 1] for operand A.
    // CHECK:         %[[VAL_1027:.*]] = llvm.getelementptr %[[VAL_1026]]{{\[}}%[[OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-COUNT-31:{{.*}} = llvm.getelementptr %[[VAL_1026]]{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         %[[offsetOuter:.*]] = llvm.mul %[[repNonKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[offsetInner:.*]] = llvm.mul %[[repKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1063:.*]] = llvm.add %[[offsetOuter]], %[[offsetInner]] : i32

    // COM: Total 32 scalar per repetition cluster [4, 1] for operand A.
    // CHECK:         %[[VAL_1064:.*]] = llvm.getelementptr %[[VAL_1027]]{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         llvm.load %[[VAL_1064]] : !llvm.ptr<3> -> f16
    // CHECK-COUNT-31:{{.*}} = llvm.getelementptr {{.*}}{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16{{[[:space:]].*}}{{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> f16

    %AA_DOT = triton_gpu.local_load %AA : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #dot_operand_a>

    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #dot_operand_b>
    %D = tt.dot %AA_DOT, %cst1, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}

// -----

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_1:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>,
  // CHECK-SAME:    %[[SCRATCH_SLM:.*]]: !llvm.ptr<3>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], {{.*}}} {
  tt.func @convert_dot(%B: tensor<64x256xf16, #blocked1>) {
    // CHECK-DAG:     %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[CST_256:.*]] = llvm.mlir.constant(256 : i32) : i32
    // CHECK-DAG:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %BB = triton_gpu.local_alloc %B : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of triton_gpu.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK:         %[[LINEAR_WARP_ID:.*]] = llvm.udiv %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[LANE_ID:.*]] = llvm.urem %[[WORK_ITEM_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[WARP_ID_N:.*]] = llvm.urem %[[LINEAR_WARP_ID]], %[[CST_8]]  : i32
    // CHECK:         %[[OUTER_WARP_ID:.*]] = llvm.urem %[[WARP_ID_N]], %[[CST_8]]  : i32
    // COM:   Compute the offsets of the elements on the SLM.
    // CHECK:         %[[repKDimStride:.*]] = llvm.mul %[[CST_16]], %[[CST_256]] : i32
    // CHECK:         %[[repNonKDimStride:.*]] = llvm.mul %[[CST_256]], %[[CST_1]] : i32
    // CHECK:         %[[warpMatStride:.*]] = llvm.mul %[[CST_32]], %[[CST_1]] : i32
    // CHECK:         %[[laneRowIndex_:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[laneRowIndex:.*]] = llvm.mul %[[laneRowIndex_]], %[[CST_2]] : i32
    // CHECK:         %[[laneColIndex:.*]] = llvm.urem %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:         %[[iOff:.*]] = llvm.mul %[[OUTER_WARP_ID]], %[[warpMatStride]] : i32
    // CHECK:         %[[rowIndex:.*]] = llvm.mul %[[CST_0]], %[[CST_2]] : i32
    // CHECK:         %[[iBase_0:.*]] = llvm.add %[[rowIndex]], %[[laneRowIndex]] : i32
    // CHECK:         %[[iBase_1:.*]] = llvm.add %[[iBase_0]], %[[CST_0]] : i32
    // CHECK:         %[[jBase_0:.*]] = llvm.add %[[laneColIndex]], %[[CST_0]] : i32
    // CHECK:         %[[iBase_Rounded:.*]] = llvm.urem %[[iBase_1]], %[[CST_64]]  : i32
    // CHECK:         %[[jBase_Rounded:.*]] = llvm.urem %[[jBase_0]], %[[CST_256]]  : i32
    // CHECK:         %[[VAL_487:.*]] = llvm.udiv %[[iBase_Rounded]], %[[CST_1]]  : i32
    // CHECK:         %[[phase:.*]] = llvm.urem %[[VAL_487]], %[[CST_1]]  : i32
    // COM: swizzle: col_swizzled = (col / vec) ^ phase * vec
    // CHECK:         %[[VAL_489:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:         %[[VAL_490:.*]] = llvm.add %[[VAL_489]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_491:.*]] = llvm.xor %[[VAL_490]], %[[phase]]  : i32
    // CHECK:         %[[jOff:.*]] = llvm.mul %[[VAL_491]], %[[CST_1]] : i32
    // CHECK:         %[[VAL_493:.*]] = llvm.mul %[[iBase_Rounded]], %[[CST_256]] : i32
    // CHECK:         %[[OFFSET_I:.*]] = llvm.add %[[VAL_493]], %[[iOff]] : i32
    // CHECK:         %[[VAL_495:.*]] = llvm.mul %[[jBase_Rounded]], %[[CST_1]] : i32
    // CHECK:         %[[OFFSET_J:.*]] = llvm.add %[[VAL_495]], %[[jOff]] : i32
    // CHECK:         %[[OFFSET:.*]] = llvm.add %[[OFFSET_I]], %[[OFFSET_J]] : i32
    // CHECK:         %[[SLM_SWIZZLE_OFFSET:.*]] = llvm.sub %[[CST_0]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1026:.*]] = llvm.getelementptr %[[SCRATCH_SLM]]{{\[}}%[[SLM_SWIZZLE_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16

    // COM: Total 32 ptrs per repetition cluster [1, 2] for operand B.
    // CHECK:         %[[VAL_1027:.*]] = llvm.getelementptr %[[VAL_1026]]{{\[}}%[[OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-COUNT-31:{{.*}} = llvm.getelementptr %[[VAL_1026]]{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         %[[offsetOuter:.*]] = llvm.mul %[[repNonKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[offsetInner:.*]] = llvm.mul %[[repKDimStride]], %[[CST_0]] : i32
    // CHECK:         %[[VAL_1063:.*]] = llvm.add %[[offsetOuter]], %[[offsetInner]] : i32

    // COM: Total 32 scalar per repetition cluster [1, 2] for operand B.
    // CHECK:         %[[VAL_1064:.*]] = llvm.getelementptr %[[VAL_1027]]{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:         llvm.load %[[VAL_1064]] : !llvm.ptr<3> -> f16
    // CHECK-COUNT-31:{{.*}} = llvm.getelementptr {{.*}}{{\[}}%[[VAL_1063]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16{{[[:space:]].*}}{{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> f16

    %BB_DOT = triton_gpu.local_load %BB : !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x256xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #dot_operand_a>

    %D = tt.dot %cst1, %BB_DOT, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}
