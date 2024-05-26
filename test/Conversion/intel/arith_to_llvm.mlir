// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// CHECK-DAG:   llvm.func spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(f32) -> i16
// CHECK-DAG: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

// CHECK-LABEL:   llvm.func spir_kernelcc @float_to_bfloat_conversion(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.struct<(f32, f32, f32, f32)>) -> !llvm.struct<(i16, i16, i16, i16)>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @float_to_bfloat_conversion(%arg0 : tensor<512xf32, #blocked>) ->  tensor<512xbf16, #blocked>{
// CHECK:                           builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.struct<(f32, f32, f32, f32)> to tensor<512xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_6:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_2]]) : (f32) -> i16
// CHECK:           %[[VAL_7:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_3]]) : (f32) -> i16
// CHECK:           %[[VAL_8:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_4]]) : (f32) -> i16
// CHECK:           %[[VAL_9:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_5]]) : (f32) -> i16
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_10]][0] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_11]][1] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_12]][2] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_13]][3] : !llvm.struct<(i16, i16, i16, i16)>
    %1 = arith.truncf %arg0 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
// CHECK:           llvm.return %[[VAL_14]] : !llvm.struct<(i16, i16, i16, i16)>
    tt.return %1: tensor<512xbf16, #blocked>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @bfloat_to_float_conversion(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.struct<(i16, i16, i16, i16)>) -> !llvm.struct<(f32, f32, f32, f32)>
  tt.func @bfloat_to_float_conversion(%arg0 : tensor<512xbf16, #blocked>) ->  tensor<512xf32, #blocked>{
// CHECK:                           builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.struct<(i16, i16, i16, i16)> to tensor<512xbf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(i16, i16, i16, i16)>
// CHECK:           %[[VAL_6:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[VAL_2]]) : (i16) -> f32
// CHECK:           %[[VAL_7:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[VAL_3]]) : (i16) -> f32
// CHECK:           %[[VAL_8:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[VAL_4]]) : (i16) -> f32
// CHECK:           %[[VAL_9:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[VAL_5]]) : (i16) -> f32
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
