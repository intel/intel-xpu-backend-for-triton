// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes { "ttg.threads-per-warp" = 16 : i32, "ttg.num-warps" = 4 : i32 } {
  // As the assert message is shared, a single instance is emitted.

  // CHECK-DAG:         llvm.mlir.global internal constant @assertFunc_("unknown\00") {addr_space = 1 : i32}
  // CHECK-DAG:         llvm.mlir.global internal constant @assertFile_("{{.*}}tritonintelgpu_to_llvm.mlir{{.*}}\00") {addr_space = 1 : i32}
  // CHECK-DAG:         llvm.mlir.global internal constant @assertMessage_("assert text\00") {addr_space = 1 : i32}
  // CHECK-DAG:         llvm.mlir.global internal constant @assertMessage_3("different assert text\00") {addr_space = 1 : i32}
  // CHECK-DAG:         llvm.func spir_funccc @__assert_fail(!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>)

  // CHECK:   llvm.func spir_kernelcc @assert(%[[VAL_0:.*]]: !llvm.struct<(i1)>, %[[VAL_1:.*]]: !llvm.struct<(i1)>, %[[VAL_2:.*]]: !llvm.struct<(i1)>, %[[PTR_1:.*]]: !llvm.ptr<1>)
  tt.func public @assert(%arg0: tensor<1xi1, #blocked>, %arg1: tensor<1xi1, #blocked>, %arg2: tensor<1xi1, #blocked>) {
    // CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(i1)>
    // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_6:.*]] = llvm.icmp "eq" %[[VAL_3]], %[[VAL_5]] : i1
    // CHECK:           %[[VAL_7:.*]] = llvm.or %[[VAL_4]], %[[VAL_6]] : i1
    // CHECK:           llvm.cond_br %[[VAL_7]], ^bb1, ^bb2
    // CHECK:         ^bb1:
    // CHECK:           %[[VAL_8:.*]] = llvm.mlir.addressof @assertMessage_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_8]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_10:.*]] = llvm.mlir.addressof @assertFile_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_10]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_12:.*]] = llvm.mlir.addressof @assertFunc_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant
    // CHECK:           %[[VAL_15:.*]] = llvm.addrspacecast %[[VAL_9]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_16:.*]] = llvm.addrspacecast %[[VAL_11]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_17:.*]] = llvm.addrspacecast %[[VAL_13]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           llvm.call spir_funccc @__assert_fail(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]], %[[VAL_17]]) : (!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>) -> ()
    // CHECK:           llvm.br ^bb2
    // CHECK:         ^bb2:
    // CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[VAL_18]]) {convergent, no_unwind, will_return} : (i32) -> ()
    tt.assert %arg0, "assert text" : tensor<1xi1, #blocked>
    // CHECK:           %[[VAL_19:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<(i1)>
    // CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_22:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_21]] : i1
    // CHECK:           %[[VAL_23:.*]] = llvm.or %[[VAL_20]], %[[VAL_22]] : i1
    // CHECK:           llvm.cond_br %[[VAL_23]], ^bb3, ^bb4
    // CHECK:         ^bb3:
    // CHECK:           %[[VAL_24:.*]] = llvm.mlir.addressof @assertMessage_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_24]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_26:.*]] = llvm.mlir.addressof @assertFile_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_26]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @assertFunc_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_28]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant
    // CHECK:           %[[VAL_31:.*]] = llvm.addrspacecast %[[VAL_25]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_32:.*]] = llvm.addrspacecast %[[VAL_27]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_33:.*]] = llvm.addrspacecast %[[VAL_29]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           llvm.call spir_funccc @__assert_fail(%[[VAL_31]], %[[VAL_32]], %[[VAL_30]], %[[VAL_33]]) : (!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>) -> ()
    // CHECK:           llvm.br ^bb4
    // CHECK:         ^bb4:
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[VAL_34]]) {convergent, no_unwind, will_return} : (i32) -> ()
    tt.assert %arg1, "assert text" : tensor<1xi1, #blocked>
    // CHECK:           %[[VAL_35:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.struct<(i1)>
    // CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:           %[[VAL_38:.*]] = llvm.icmp "eq" %[[VAL_35]], %[[VAL_37]] : i1
    // CHECK:           %[[VAL_39:.*]] = llvm.or %[[VAL_36]], %[[VAL_38]] : i1
    // CHECK:           llvm.cond_br %[[VAL_39]], ^bb5, ^bb6
    // CHECK:         ^bb5:
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.addressof @assertMessage_3 : !llvm.ptr<1>
    // CHECK:           %[[VAL_41:.*]] = llvm.getelementptr %[[VAL_40]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_42:.*]] = llvm.mlir.addressof @assertFile_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_43:.*]] = llvm.getelementptr %[[VAL_42]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_44:.*]] = llvm.mlir.addressof @assertFunc_ : !llvm.ptr<1>
    // CHECK:           %[[VAL_45:.*]] = llvm.getelementptr %[[VAL_44]][0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant
    // CHECK:           %[[VAL_47:.*]] = llvm.addrspacecast %[[VAL_41]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_48:.*]] = llvm.addrspacecast %[[VAL_43]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           %[[VAL_49:.*]] = llvm.addrspacecast %[[VAL_45]] : !llvm.ptr<1> to !llvm.ptr<4>
    // CHECK:           llvm.call spir_funccc @__assert_fail(%[[VAL_47]], %[[VAL_48]], %[[VAL_46]], %[[VAL_49]]) : (!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>) -> ()
    // CHECK:           llvm.br ^bb6
    // CHECK:         ^bb6:
    // CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[VAL_50]]) {convergent, no_unwind, will_return} : (i32) -> ()
    tt.assert %arg2, "different assert text" : tensor<1xi1, #blocked>
    tt.return
  }
}

// -----

// Sanity check for the conversion pass to correctly process even empty modules
module attributes { "ttg.threads-per-warp" = 16 : i32, "ttg.num-warps" = 4 : i32 } {}
