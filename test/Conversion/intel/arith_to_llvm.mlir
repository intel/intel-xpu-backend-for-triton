// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// CHECK-DAG: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

// CHECK-LABEL:   llvm.func spir_kernelcc @float_to_bfloat_conversion(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.struct<(f32, f32, f32, f32)>) -> !llvm.struct<(bf16, bf16, bf16, bf16)>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @float_to_bfloat_conversion(%arg0 : tensor<512xf32, #blocked>) ->  tensor<512xbf16, #blocked>{
// CHECK:                           builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.struct<(f32, f32, f32, f32)> to tensor<512xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_6:.*]] = llvm.call spir_funccc @_Z32intel_convert_bfloat16_as_ushortf(%[[VAL_2]]) : (f32) -> i16
// CHECK:           %[[BITCAST_6:.*]] = llvm.bitcast %[[VAL_6]] : i16 to bf16
// CHECK:           %[[VAL_7:.*]] = llvm.call spir_funccc @_Z32intel_convert_bfloat16_as_ushortf(%[[VAL_3]]) : (f32) -> i16
// CHECK:           %[[BITCAST_7:.*]] = llvm.bitcast %[[VAL_7]] : i16 to bf16
// CHECK:           %[[VAL_8:.*]] = llvm.call spir_funccc @_Z32intel_convert_bfloat16_as_ushortf(%[[VAL_4]]) : (f32) -> i16
// CHECK:           %[[BITCAST_8:.*]] = llvm.bitcast %[[VAL_8]] : i16 to bf16
// CHECK:           %[[VAL_9:.*]] = llvm.call spir_funccc @_Z32intel_convert_bfloat16_as_ushortf(%[[VAL_5]]) : (f32) -> i16
// CHECK:           %[[BITCAST_9:.*]] = llvm.bitcast %[[VAL_9]] : i16 to bf16
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[BITCAST_6]], %[[VAL_10]][0] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[BITCAST_7]], %[[VAL_11]][1] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[BITCAST_8]], %[[VAL_12]][2] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[BITCAST_9]], %[[VAL_13]][3] : !llvm.struct<(bf16, bf16, bf16, bf16)>
    %1 = arith.truncf %arg0 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
// CHECK:           llvm.return %[[VAL_14]] : !llvm.struct<(bf16, bf16, bf16, bf16)>
    tt.return %1: tensor<512xbf16, #blocked>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @bfloat_to_float_conversion(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.struct<(bf16, bf16, bf16, bf16)>) -> !llvm.struct<(f32, f32, f32, f32)>
  tt.func @bfloat_to_float_conversion(%arg0 : tensor<512xbf16, #blocked>) ->  tensor<512xf32, #blocked>{
// CHECK:                           builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.struct<(bf16, bf16, bf16, bf16)> to tensor<512xbf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK:           %[[BITCAST_2:.*]] = llvm.bitcast %[[VAL_2]] : bf16 to i16
// CHECK:           %[[VAL_6:.*]] = llvm.call spir_funccc @_Z31intel_convert_as_bfloat16_floatt(%[[BITCAST_2]]) : (i16) -> f32
// CHECK:           %[[BITCAST_3:.*]] = llvm.bitcast %[[VAL_3]] : bf16 to i16
// CHECK:           %[[VAL_7:.*]] = llvm.call spir_funccc @_Z31intel_convert_as_bfloat16_floatt(%[[BITCAST_3]]) : (i16) -> f32
// CHECK:           %[[BITCAST_4:.*]] = llvm.bitcast %[[VAL_4]] : bf16 to i16
// CHECK:           %[[VAL_8:.*]] = llvm.call spir_funccc @_Z31intel_convert_as_bfloat16_floatt(%[[BITCAST_4]]) : (i16) -> f32
// CHECK:           %[[BITCAST_5:.*]] = llvm.bitcast %[[VAL_5]] : bf16 to i16
// CHECK:           %[[VAL_9:.*]] = llvm.call spir_funccc @_Z31intel_convert_as_bfloat16_floatt(%[[BITCAST_5]]) : (i16) -> f32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_10]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_11]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_12]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_13]][3] : !llvm.struct<(f32, f32, f32, f32)>
    %1 = arith.extf %arg0 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
// CHECK:           llvm.return %[[VAL_14]] : !llvm.struct<(f32, f32, f32, f32)>
    tt.return %1: tensor<512xf32, #blocked>
  }
}
