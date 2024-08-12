// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_mma_layout_emit_off()
  tt.func public @test_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32x32xi32, #mma>
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:           %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:           %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:           %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32

    // CHECK:           %[[THREAD_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:           %[[THREAD_ID_32:.*]] = llvm.trunc %[[THREAD_ID]] : i64 to i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[THREAD_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[THREAD_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[WARP_ID_Y:.*]] = llvm.urem %[[WARP_ID]], %[[CST_2]]  : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.udiv %[[WARP_ID]], %[[CST_2]]  : i32
    // CHECK:           %[[WARP_ID_X:.*]] = llvm.urem %[[VAL_23]], %[[CST_2]]  : i32
    // CHECK:           %[[ROUNDED_WARP_ID_X:.*]] = llvm.urem %[[WARP_ID_X]], %[[CST_4]]  : i32
    // CHECK:           %[[ROUNDED_WARP_ID_Y:.*]] = llvm.urem %[[WARP_ID_Y]], %[[CST_4]]  : i32
    // CHECK:           %[[WARP_OFFSET_X:.*]] = llvm.mul %[[ROUNDED_WARP_ID_X]], %[[CST_8]] : i32
    // CHECK:           %[[WARP_OFFSET_Y:.*]] = llvm.mul %[[ROUNDED_WARP_ID_Y]], %[[CST_8]] : i32
    // CHECK:           %[[LANE_OFFSET_X:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_8]]  : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[LANE_OFFSET_X]], %[[WARP_OFFSET_X]] : i32
    // CHECK:           %[[LANE_OFFSET_Y:.*]] = llvm.urem %[[LANE_ID]], %[[CST_8]]  : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[LANE_OFFSET_Y]], %[[WARP_OFFSET_Y]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.urem %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.urem %[[VAL_34]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.urem %[[VAL_35]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.urem %[[VAL_33]], %[[CST_1]]  : i32
    // CHECK:           %[[CTA_OFFSET_Y:.*]] = llvm.mul %[[VAL_36]], %[[CST_32]] : i32
    // CHECK:           %[[CTA_OFFSET_X:.*]] = llvm.mul %[[VAL_37]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.add %[[OFFSET_X]], %[[CTA_OFFSET_Y]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.add %[[OFFSET_Y]], %[[CTA_OFFSET_X]] : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.add %[[VAL_40]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_0:.*]] = llvm.add %[[VAL_41]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.add %[[VAL_40]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.add %[[VAL_40]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.add %[[VAL_40]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_Y_1:.*]] = llvm.add %[[VAL_41]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.add %[[VAL_40]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.add %[[VAL_40]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.add %[[VAL_40]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.add %[[VAL_40]], %[[CST_22]] : i32
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32x32xi32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
#dot_op_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_dot_mma_layout_emit_off()
  tt.func public @test_dot_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32x32xi32, #dot_op_a>
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:           %[[CST_5:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG:           %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:           %[[CST_7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG:           %[[CST_17:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK-DAG:           %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:           %[[CST_19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:           %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:           %[[CST_21:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK-DAG:           %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK-DAG:           %[[CST_23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK:           %[[THREAD_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:           %[[THREAD_ID_32:.*]] = llvm.trunc %[[THREAD_ID]] : i64 to i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[THREAD_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[THREAD_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.udiv %[[WARP_ID]], %[[CST_2]]  : i32
    // CHECK:           %[[WARP_ID_X:.*]] = llvm.urem %[[VAL_29]], %[[CST_2]]  : i32
    // CHECK:           %[[ROUNDED_WARP_ID_X:.*]] = llvm.urem %[[WARP_ID_X]], %[[CST_4]]  : i32
    // CHECK:           %[[WARP_OFFSET:.*]] = llvm.mul %[[ROUNDED_WARP_ID_X]], %[[CST_8]] : i32
    // CHECK:           %[[LANE_ID_X:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:           %[[LANE_ID_Y:.*]] = llvm.urem %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.mul %[[LANE_ID_Y]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_x:.*]] = llvm.add %[[LANE_ID_X]], %[[WARP_OFFSET]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.urem %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.urem %[[VAL_38]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.urem %[[VAL_39]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.urem %[[VAL_37]], %[[CST_1]]  : i32
    // CHECK:           %[[CTA_OFFSET_X:.*]] = llvm.mul %[[VAL_40]], %[[CST_32]] : i32
    // CHECK:           %[[CTA_OFFSET_Y:.*]] = llvm.mul %[[VAL_41]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_44:.*]] = llvm.add %[[OFFSET_x]], %[[CTA_OFFSET_X]] : i32
    // CHECK:           %[[VAL_45:.*]] = llvm.add %[[OFFSET_Y]], %[[CTA_OFFSET_Y]] : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.add %[[VAL_44]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_0:.*]] = llvm.add %[[VAL_45]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_1:.*]] = llvm.add %[[VAL_45]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.add %[[VAL_44]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.add %[[VAL_44]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.add %[[VAL_44]], %[[CST_3]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.add %[[VAL_44]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.add %[[VAL_44]], %[[CST_5]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.add %[[VAL_44]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.add %[[VAL_44]], %[[CST_7]] : i32
    // CHECK:           %[[OFFSET_X_8:.*]] = llvm.add %[[VAL_44]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_9:.*]] = llvm.add %[[VAL_44]], %[[CST_17]] : i32
    // CHECK:           %[[OFFSET_X_10:.*]] = llvm.add %[[VAL_44]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_11:.*]] = llvm.add %[[VAL_44]], %[[CST_19]] : i32
    // CHECK:           %[[OFFSET_X_12:.*]] = llvm.add %[[VAL_44]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_13:.*]] = llvm.add %[[VAL_44]], %[[CST_21]] : i32
    // CHECK:           %[[OFFSET_X_14:.*]] = llvm.add %[[VAL_44]], %[[CST_22]] : i32
    // CHECK:           %[[OFFSET_X_15:.*]] = llvm.add %[[VAL_44]], %[[CST_23]] : i32
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32x32xi32, #dot_op_a>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
#dot_op_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#slice = #triton_gpu.slice<{dim = 1, parent = #dot_op_a}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_slice_dot_mma_layout_emit_off()
  tt.func public @test_slice_dot_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32xi32, #slice>
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:           %[[CST_5:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG:           %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:           %[[CST_7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG:           %[[CST_17:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK-DAG:           %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:           %[[CST_19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:           %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:           %[[CST_21:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK-DAG:           %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK-DAG:           %[[CST_23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK:           %[[THREADS_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:           %[[THREADS_ID_32:.*]] = llvm.trunc %[[THREADS_ID]] : i64 to i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[THREADS_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[THREADS_ID_32]], %[[CST_16]]  : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.udiv %[[WARP_ID]], %[[CST_2]]  : i32
    // CHECK:           %[[WARP_ID_X:.*]] = llvm.urem %[[VAL_29]], %[[CST_2]]  : i32
    // CHECK:           %[[ROUNDED_WARP_ID_X:.*]] = llvm.urem %[[WARP_ID_X]], %[[CST_4]]  : i32
    // CHECK:           %[[WARP_OFFSET_X:.*]] = llvm.mul %[[ROUNDED_WARP_ID_X]], %[[CST_8]] : i32
    // CHECK:           %[[LANE_OFFSET_X:.*]] = llvm.udiv %[[LANE_ID]], %[[CST_16]]  : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[LANE_OFFSET_X]], %[[WARP_OFFSET_X]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.udiv %[[CST_0]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.urem %[[VAL_35]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.urem %[[VAL_36]], %[[CST_1]]  : i32
    // CHECK:           %[[CTA_OFFSET_X:.*]] = llvm.mul %[[VAL_37]], %[[CST_32]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.add %[[OFFSET_X]], %[[CTA_OFFSET_X]] : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.add %[[VAL_39]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.add %[[VAL_39]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.add %[[VAL_39]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.add %[[VAL_39]], %[[CST_3]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.add %[[VAL_39]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.add %[[VAL_39]], %[[CST_5]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.add %[[VAL_39]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.add %[[VAL_39]], %[[CST_7]] : i32
    // CHECK:           %[[OFFSET_X_8:.*]] = llvm.add %[[VAL_39]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_9:.*]] = llvm.add %[[VAL_39]], %[[CST_17]] : i32
    // CHECK:           %[[OFFSET_X_10:.*]] = llvm.add %[[VAL_39]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_11:.*]] = llvm.add %[[VAL_39]], %[[CST_19]] : i32
    // CHECK:           %[[OFFSET_X_12:.*]] = llvm.add %[[VAL_39]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_13:.*]] = llvm.add %[[VAL_39]], %[[CST_21]] : i32
    // CHECK:           %[[OFFSET_X_14:.*]] = llvm.add %[[VAL_39]], %[[CST_22]] : i32
    // CHECK:           %[[OFFSET_X_15:.*]] = llvm.add %[[VAL_39]], %[[CST_23]] : i32
    // CHECK:           %[[VAL_56:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_57:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_58:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_59:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_60:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_61:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_62:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_63:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_64:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_65:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_66:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_67:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_68:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_69:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_70:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_71:.*]] = llvm.call @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32xi32, #slice>
    tt.return
  }
}
