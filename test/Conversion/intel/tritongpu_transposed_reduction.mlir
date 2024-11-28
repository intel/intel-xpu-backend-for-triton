// RUN: env TRITON_INTEL_ADVANCED_PATH=1 TRITON_INTEL_REDUCE_TRANSPOSE=1 \
// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Checks the correct lowering of transpose reductions.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK:         llvm.func spir_funccc @[[MAXNUM:.*]](
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) -> f32 attributes {always_inline} {
// CHECK:           %[[VAL_2:.*]] = llvm.intr.maxnum(%[[VAL_0]], %[[VAL_1]])  {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32) -> f32
// CHECK:           llvm.return %[[VAL_2]] : f32
// CHECK:         }

// CHECK:         llvm.func spir_funccc @[[ADD:.*]](
// CHECK-SAME:                                      %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) -> f32 attributes {always_inline} {
// CHECK:           %[[VAL_2:.*]] = llvm.fadd %[[VAL_0]], %[[VAL_1]]  {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK:           llvm.return %[[VAL_2]] : f32
// CHECK:         }

// CHECK:         llvm.func spir_kernelcc @reduce_sum(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<16xf32>) -> f32 attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
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
// CHECK:           %[[VAL_34:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_3]], %[[VAL_5]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_35:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_7]], %[[VAL_9]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_11]], %[[VAL_13]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_37:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_15]], %[[VAL_17]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_19]], %[[VAL_21]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_39:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_23]], %[[VAL_25]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_40:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_27]], %[[VAL_29]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_41:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_31]], %[[VAL_33]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_42:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_34]], %[[VAL_35]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_43:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_36]], %[[VAL_37]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_44:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_38]], %[[VAL_39]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_45:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_40]], %[[VAL_41]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_46:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_42]], %[[VAL_43]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_47:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_44]], %[[VAL_45]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_48:.*]] = llvm.call spir_funccc @[[ADD]](%[[VAL_46]], %[[VAL_47]]) : (f32, f32) -> f32
// CHECK:           llvm.return %[[VAL_48]] : f32
// CHECK:         }
  tt.func public @reduce_sum(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    tt.return %0: tensor<16xf32>
  }

// CHECK:         llvm.func spir_kernelcc @reduce_max(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<16xf32>) -> f32 attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
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
// CHECK:           %[[VAL_34:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_3]], %[[VAL_5]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_35:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_7]], %[[VAL_9]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_11]], %[[VAL_13]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_37:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_15]], %[[VAL_17]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_19]], %[[VAL_21]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_39:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_23]], %[[VAL_25]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_40:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_27]], %[[VAL_29]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_41:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_31]], %[[VAL_33]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_42:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_34]], %[[VAL_35]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_43:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_36]], %[[VAL_37]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_44:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_38]], %[[VAL_39]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_45:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_40]], %[[VAL_41]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_46:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_42]], %[[VAL_43]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_47:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_44]], %[[VAL_45]]) : (f32, f32) -> f32
// CHECK:           %[[VAL_48:.*]] = llvm.call spir_funccc @[[MAXNUM]](%[[VAL_46]], %[[VAL_47]]) : (f32, f32) -> f32
// CHECK:           llvm.return %[[VAL_48]] : f32
// CHECK:         }
  tt.func public @reduce_max(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    tt.return %0: tensor<16xf32>
  }
}
