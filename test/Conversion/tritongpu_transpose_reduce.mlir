// RUN: TRITON_INTEL_ENABLE_BLOCK_PTR=1 triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {

// Reduce operation

// CHECK-LABEL:   llvm.func spir_funccc @__reduceOp_
// CHECK-SAME:                                      [[REDUCE_FUNC_SUFFIX:.*]](
// CHECK-SAME:                                                                %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                                                %[[VAL_1:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_2:.*]] = llvm.intr.maxnum(%[[VAL_0]], %[[VAL_1]])  : (f32, f32) -> f32
// CHECK:           llvm.return %[[VAL_2]] : f32
// CHECK:         }

// CHECK:         llvm.func spir_funccc @_Z22get_sub_group_local_idv() -> i32
// CHECK:         llvm.func spir_funccc @_Z16get_sub_group_idv() -> i32

// CHECK-LABEL:   llvm.func spir_kernelcc @sub_group_transpose(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !llvm.ptr<3>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: vector<16xf32>) attributes {passthrough = {{\[\[}}"gen.intel_reqd_sub_group_size", "16"], ["gen.max_work_group_size", "512,1,1"]]} {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.ptr<3> to !tt.ptr<f32, 3>
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z16get_sub_group_idv()
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_idv()
// CHECK:           %[[VAL_7:.*]] = llvm.mul %[[VAL_5]], %[[VAL_4]]  : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_6]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_10]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_11]], %[[VAL_9]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_9]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_13]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_14]], %[[VAL_12]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_12]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_16]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_17]], %[[VAL_15]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_15]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_19]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_20]], %[[VAL_18]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_18]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_22]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_23]], %[[VAL_21]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_21]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_26:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_25]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_26]], %[[VAL_24]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_24]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_28]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_29]], %[[VAL_27]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_30:.*]] = llvm.getelementptr %[[VAL_27]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_32:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_31]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_32]], %[[VAL_30]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_33:.*]] = llvm.getelementptr %[[VAL_30]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_34]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_35]], %[[VAL_33]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_33]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_38:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_37]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_38]], %[[VAL_36]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_39:.*]] = llvm.getelementptr %[[VAL_36]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_41:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_40]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_41]], %[[VAL_39]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_42:.*]] = llvm.getelementptr %[[VAL_39]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(11 : i32) : i32
// CHECK:           %[[VAL_44:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_43]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_44]], %[[VAL_42]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_45:.*]] = llvm.getelementptr %[[VAL_42]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_46]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_47]], %[[VAL_45]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_48:.*]] = llvm.getelementptr %[[VAL_45]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_50:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_49]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_50]], %[[VAL_48]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_51:.*]] = llvm.getelementptr %[[VAL_48]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_52:.*]] = llvm.mlir.constant(14 : i32) : i32
// CHECK:           %[[VAL_53:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_52]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_53]], %[[VAL_51]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_54:.*]] = llvm.getelementptr %[[VAL_51]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_55:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_56:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_55]] : i32] : vector<16xf32>
// CHECK:           llvm.store %[[VAL_56]], %[[VAL_54]] : f32, !llvm.ptr<3>
// CHECK:           %[[VAL_57:.*]] = llvm.getelementptr %[[VAL_54]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_58:.*]] = llvm.mlir.poison : vector<16xf32>
// CHECK:           %[[VAL_59:.*]] = llvm.mul %[[VAL_5]], %[[VAL_6]]  : i32
// CHECK:           %[[VAL_60:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_59]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_61:.*]] = llvm.load %[[VAL_60]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_62:.*]] = llvm.getelementptr %[[VAL_60]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_63:.*]] = llvm.load %[[VAL_62]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_64:.*]] = llvm.getelementptr %[[VAL_62]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_65:.*]] = llvm.load %[[VAL_64]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_66:.*]] = llvm.getelementptr %[[VAL_64]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_67:.*]] = llvm.load %[[VAL_66]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_68:.*]] = llvm.getelementptr %[[VAL_66]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_69:.*]] = llvm.load %[[VAL_68]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_70:.*]] = llvm.getelementptr %[[VAL_68]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_71:.*]] = llvm.load %[[VAL_70]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_72:.*]] = llvm.getelementptr %[[VAL_70]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_73:.*]] = llvm.load %[[VAL_72]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_74:.*]] = llvm.getelementptr %[[VAL_72]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_75:.*]] = llvm.load %[[VAL_74]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_76:.*]] = llvm.getelementptr %[[VAL_74]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_77:.*]] = llvm.load %[[VAL_76]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_78:.*]] = llvm.getelementptr %[[VAL_76]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_79:.*]] = llvm.load %[[VAL_78]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_80:.*]] = llvm.getelementptr %[[VAL_78]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_81:.*]] = llvm.load %[[VAL_80]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_82:.*]] = llvm.getelementptr %[[VAL_80]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_83:.*]] = llvm.load %[[VAL_82]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_84:.*]] = llvm.getelementptr %[[VAL_82]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_85:.*]] = llvm.load %[[VAL_84]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_86:.*]] = llvm.getelementptr %[[VAL_84]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_87:.*]] = llvm.load %[[VAL_86]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_88:.*]] = llvm.getelementptr %[[VAL_86]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_89:.*]] = llvm.load %[[VAL_88]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_90:.*]] = llvm.getelementptr %[[VAL_88]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_91:.*]] = llvm.load %[[VAL_90]] : !llvm.ptr<3> -> f32
// CHECK:           %[[VAL_92:.*]] = llvm.getelementptr %[[VAL_90]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_93:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_94:.*]] = llvm.insertelement %[[VAL_61]], %[[VAL_58]]{{\[}}%[[VAL_93]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_95:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_96:.*]] = llvm.insertelement %[[VAL_63]], %[[VAL_94]]{{\[}}%[[VAL_95]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_97:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_98:.*]] = llvm.insertelement %[[VAL_65]], %[[VAL_96]]{{\[}}%[[VAL_97]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_99:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_100:.*]] = llvm.insertelement %[[VAL_67]], %[[VAL_98]]{{\[}}%[[VAL_99]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_101:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_102:.*]] = llvm.insertelement %[[VAL_69]], %[[VAL_100]]{{\[}}%[[VAL_101]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_103:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_104:.*]] = llvm.insertelement %[[VAL_71]], %[[VAL_102]]{{\[}}%[[VAL_103]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_105:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_106:.*]] = llvm.insertelement %[[VAL_73]], %[[VAL_104]]{{\[}}%[[VAL_105]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_107:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_108:.*]] = llvm.insertelement %[[VAL_75]], %[[VAL_106]]{{\[}}%[[VAL_107]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_109:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_110:.*]] = llvm.insertelement %[[VAL_77]], %[[VAL_108]]{{\[}}%[[VAL_109]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_111:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_112:.*]] = llvm.insertelement %[[VAL_79]], %[[VAL_110]]{{\[}}%[[VAL_111]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_113:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_114:.*]] = llvm.insertelement %[[VAL_81]], %[[VAL_112]]{{\[}}%[[VAL_113]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_115:.*]] = llvm.mlir.constant(11 : i32) : i32
// CHECK:           %[[VAL_116:.*]] = llvm.insertelement %[[VAL_83]], %[[VAL_114]]{{\[}}%[[VAL_115]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_117:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_118:.*]] = llvm.insertelement %[[VAL_85]], %[[VAL_116]]{{\[}}%[[VAL_117]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_119:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_120:.*]] = llvm.insertelement %[[VAL_87]], %[[VAL_118]]{{\[}}%[[VAL_119]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_121:.*]] = llvm.mlir.constant(14 : i32) : i32
// CHECK:           %[[VAL_122:.*]] = llvm.insertelement %[[VAL_89]], %[[VAL_120]]{{\[}}%[[VAL_121]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_123:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_124:.*]] = llvm.insertelement %[[VAL_91]], %[[VAL_122]]{{\[}}%[[VAL_123]] : i32] : vector<16xf32>
// CHECK:           llvm.return
// CHECK:         }

  tt.func public @sub_group_transpose(%arg0: !tt.ptr<f32, 3>, %arg1: tensor<16x16xf32>) {
    %res = triton_intel_gpu.sub_group_transpose %arg0, %arg1 : (!tt.ptr<f32, 3>, tensor<16x16xf32>) -> tensor<16x16xf32>
    tt.return
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @sub_group_reduce(
// CHECK-SAME:                                              %[[VAL_0:.*]]: vector<16xf32>) -> f32 attributes {passthrough = {{\[\[}}"gen.intel_reqd_sub_group_size", "16"], ["gen.max_work_group_size", "512,1,1"]]} {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_2]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_4]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_6]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_8]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_10]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_12]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_14]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_16]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_18]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_21:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_20]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_22]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(11 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_24]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_26]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_28]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(14 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_30]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_32]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_34:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_3]], %[[VAL_19]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_35:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_5]], %[[VAL_21]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_36:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_7]], %[[VAL_23]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_37:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_9]], %[[VAL_25]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_38:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_11]], %[[VAL_27]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_39:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_13]], %[[VAL_29]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_40:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_15]], %[[VAL_31]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_41:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_17]], %[[VAL_33]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_42:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_34]], %[[VAL_38]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_43:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_35]], %[[VAL_39]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_44:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_36]], %[[VAL_40]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_45:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_37]], %[[VAL_41]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_46:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_42]], %[[VAL_44]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_47:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_43]], %[[VAL_45]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_48:.*]] = llvm.call @__reduceOp_[[REDUCE_FUNC_SUFFIX]](%[[VAL_46]], %[[VAL_47]]) : (f32, f32) -> f32
// CHECK:           llvm.return %[[VAL_48]] : f32

  tt.func public @sub_group_reduce(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.maxnumf %arg1, %arg2 : f32
        tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    tt.return %0 : tensor<16xf32>
  }
}
