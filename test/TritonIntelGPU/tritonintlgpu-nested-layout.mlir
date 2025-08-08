// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_mma_layout_emit_off(
  // CHECK-SAME:    %[[PTR_1:.*]]: !llvm.ptr<1>)
  tt.func public @test_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32x32xi32, #mma>
    // CHECK-DAG:       %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:       %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:       %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:       %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:       %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       %[[CST_63:.*]] = llvm.mlir.constant(63 : i32) : i32
    // CHECK-DAG:       %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:       %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:       %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:       %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK-DAG:       %[[THREAD_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK-DAG:       %[[THREAD_ID_32:.*]] = llvm.trunc %[[THREAD_ID]] : i64 to i32
    // CHECK-DAG:       %[[RTID:.*]] = llvm.and %[[THREAD_ID_32:.*]], %[[CST_63]] : i32
    // CHECK-DAG:       %[[LANE_ID:.*]] = llvm.urem %[[RTID]], %[[CST_16]] : i32
    // CHECK-DAG:       %[[WARP_ID:.*]] = llvm.udiv %[[RTID]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.and %[[WARP_ID]], %[[CST_1]]  : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.icmp "eq" %[[VAL_37]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.select %[[VAL_38]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_40:.*]] = llvm.xor %{{.*}}, %[[VAL_39]]  : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.and %[[WARP_ID]], %[[CST_2]]  : i32
    // CHECK:           %[[VAL_42:.*]] = llvm.icmp "eq" %[[VAL_41]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.select %[[VAL_42]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_44:.*]] = llvm.xor %{{.*}}, %[[VAL_43]]  : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.xor %[[VAL_44]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_0:.*]] = llvm.xor %[[VAL_40]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.xor %[[VAL_44]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.xor %[[VAL_44]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.xor %[[VAL_44]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_Y_1:.*]] = llvm.xor %[[VAL_40]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.xor %[[VAL_44]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.xor %[[VAL_44]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.xor %[[VAL_44]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.xor %[[VAL_44]], %[[CST_22]] : i32
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32x32xi32, #mma>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_dot_mma_layout_emit_off(
  // CHECK-SAME:    %[[PTR_1:.*]]: !llvm.ptr<1>)
  tt.func public @test_dot_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32x32xi32, #dot_op_a>
    // CHECK-DAG:       %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:       %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:       %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:       %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:       %[[CST_5:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG:       %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:       %[[CST_7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG:       %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:       %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       %[[CST_17:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK-DAG:       %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:       %[[CST_19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:       %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:       %[[CST_21:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK-DAG:       %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK-DAG:       %[[CST_23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK-DAG:       %[[CST_63:.*]] = llvm.mlir.constant(63 : i32) : i32
    // CHECK:           %[[THREAD_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:           %[[THREAD_ID_32:.*]] = llvm.trunc %[[THREAD_ID]] : i64 to i32
    // CHECK:           %[[RTID:.*]] = llvm.and %[[THREAD_ID_32:.*]], %[[CST_63]] : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[RTID]], %[[CST_16]]  : i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[RTID]], %[[CST_16]]  : i32
    // CHECK:           %[[VAL_27:.*]] = llvm.and %[[LANE_ID]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_28:.*]] = llvm.icmp "eq" %[[VAL_27]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.select %[[VAL_28]], %[[CST_0]], %[[CST_2]] : i1, i32
    // CHECK:           %[[VAL_30:.*]] = llvm.xor %[[CST_0]], %[[VAL_29]] : i32
    // CHECK:           %[[VAL_31:.*]] = llvm.and %[[LANE_ID]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_32:.*]] = llvm.icmp "eq" %[[VAL_31]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.select %[[VAL_32]], %[[CST_0]], %[[CST_4]] : i1, i32
    // CHECK:           %[[VAL_34:.*]] = llvm.xor %[[VAL_30]], %[[VAL_33]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.and %[[LANE_ID]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.icmp "eq" %[[VAL_35]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.select %[[VAL_36]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_38:.*]] = llvm.xor %[[VAL_34]], %[[VAL_37]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.and %[[LANE_ID]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.icmp "eq" %[[VAL_39]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.select %[[VAL_40]], %[[CST_0]], %[[CST_16]] : i1, i32
    // CHECK:           %[[VAL_42:.*]] = llvm.xor %[[VAL_38]], %[[VAL_41]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.and %[[WARP_ID]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_44:.*]] = llvm.icmp "eq" %[[VAL_43]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_45:.*]] = llvm.select %[[VAL_44]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_46:.*]] = llvm.xor %[[CST_0]], %[[VAL_45]] : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.xor %[[VAL_46]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_0:.*]] = llvm.xor %[[VAL_42]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_Y_1:.*]] = llvm.xor %[[VAL_42]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.xor %[[VAL_46]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.xor %[[VAL_46]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.xor %[[VAL_46]], %[[CST_3]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.xor %[[VAL_46]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.xor %[[VAL_46]], %[[CST_5]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.xor %[[VAL_46]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.xor %[[VAL_46]], %[[CST_7]] : i32
    // CHECK:           %[[OFFSET_X_8:.*]] = llvm.xor %[[VAL_46]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_9:.*]] = llvm.xor %[[VAL_46]], %[[CST_17]] : i32
    // CHECK:           %[[OFFSET_X_10:.*]] = llvm.xor %[[VAL_46]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_11:.*]] = llvm.xor %[[VAL_46]], %[[CST_19]] : i32
    // CHECK:           %[[OFFSET_X_12:.*]] = llvm.xor %[[VAL_46]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_13:.*]] = llvm.xor %[[VAL_46]], %[[CST_21]] : i32
    // CHECK:           %[[OFFSET_X_14:.*]] = llvm.xor %[[VAL_46]], %[[CST_22]] : i32
    // CHECK:           %[[OFFSET_X_15:.*]] = llvm.xor %[[VAL_46]], %[[CST_23]] : i32
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], %[[OFFSET_Y_0]], {{.*}}, {{.*}})
    // CHECK:           llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], %[[OFFSET_Y_1]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32x32xi32, #dot_op_a>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 32], B = [32, 8], C = [8, 8]}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#slice = #ttg.slice<{dim = 1, parent = #dot_op_a}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_slice_dot_mma_layout_emit_off(
  // CHECK-SAME:    %[[PTR_1:.*]]: !llvm.ptr<1>)
  tt.func public @test_slice_dot_mma_layout_emit_off() {
    %cst = arith.constant dense<4> : tensor<32xi32, #slice>
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_5:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG:           %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG:           %[[CST_7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_17:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK-DAG:           %[[CST_18:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK-DAG:           %[[CST_19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:           %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK-DAG:           %[[CST_21:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK-DAG:           %[[CST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK-DAG:           %[[CST_23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.xor {{.*}} : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.xor %[[CST_0]], %[[VAL_34]] : i32
    // CHECK:           %[[OFFSET_X_0:.*]] = llvm.xor %[[VAL_35]], %[[CST_0]] : i32
    // CHECK:           %[[OFFSET_X_1:.*]] = llvm.xor %[[VAL_35]], %[[CST_1]] : i32
    // CHECK:           %[[OFFSET_X_2:.*]] = llvm.xor %[[VAL_35]], %[[CST_2]] : i32
    // CHECK:           %[[OFFSET_X_3:.*]] = llvm.xor %[[VAL_35]], %[[CST_3]] : i32
    // CHECK:           %[[OFFSET_X_4:.*]] = llvm.xor %[[VAL_35]], %[[CST_4]] : i32
    // CHECK:           %[[OFFSET_X_5:.*]] = llvm.xor %[[VAL_35]], %[[CST_5]] : i32
    // CHECK:           %[[OFFSET_X_6:.*]] = llvm.xor %[[VAL_35]], %[[CST_6]] : i32
    // CHECK:           %[[OFFSET_X_7:.*]] = llvm.xor %[[VAL_35]], %[[CST_7]] : i32
    // CHECK:           %[[OFFSET_X_8:.*]] = llvm.xor %[[VAL_35]], %[[CST_16]] : i32
    // CHECK:           %[[OFFSET_X_9:.*]] = llvm.xor %[[VAL_35]], %[[CST_17]] : i32
    // CHECK:           %[[OFFSET_X_10:.*]] = llvm.xor %[[VAL_35]], %[[CST_18]] : i32
    // CHECK:           %[[OFFSET_X_11:.*]] = llvm.xor %[[VAL_35]], %[[CST_19]] : i32
    // CHECK:           %[[OFFSET_X_12:.*]] = llvm.xor %[[VAL_35]], %[[CST_20]] : i32
    // CHECK:           %[[OFFSET_X_13:.*]] = llvm.xor %[[VAL_35]], %[[CST_21]] : i32
    // CHECK:           %[[OFFSET_X_14:.*]] = llvm.xor %[[VAL_35]], %[[CST_22]] : i32
    // CHECK:           %[[OFFSET_X_15:.*]] = llvm.xor %[[VAL_35]], %[[CST_23]] : i32
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_0]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_57:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_1]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_58:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_2]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_59:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_3]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_60:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_4]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_61:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_5]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_62:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_6]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_63:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_7]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_64:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_8]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_65:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_9]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_66:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_10]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_67:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_11]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_68:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_12]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_69:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_13]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_70:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_14]], {{.*}}, {{.*}})
    // CHECK:           %[[VAL_71:.*]] = llvm.call spir_funccc @_Z18__spirv_ocl_printf({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X_15]], {{.*}}, {{.*}})
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %cst : tensor<32xi32, #slice>
    tt.return
  }
}
