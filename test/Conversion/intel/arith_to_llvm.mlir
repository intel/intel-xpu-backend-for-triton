// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

// CHECK-SCALAR-DAG:   llvm.func spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(i16) -> f32
// CHECK-SCALAR-DAG:   llvm.func spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(f32) -> i16

// CHECK-LABEL:   llvm.func spir_kernelcc @float_to_bfloat_conversion(
// CHECK-SCALAR:                                             %[[VAL_0:.*]]: !llvm.struct<(f32, f32, f32, f32)>) -> !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-VECTOR:                                             %[[VAL_0:.*]]: vector<32xf32>) -> vector<32xbf16>
module attributes {"triton_intel_gpu.support_sg_2d_block", "triton_intel_gpu.support_dpas", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @float_to_bfloat_conversion(%arg0 : tensor<512xf32, #blocked>) ->  tensor<512xbf16, #blocked>{
// CHECK-SCALAR:    %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_6:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_2]]) : (f32) -> i16
// CHECK-SCALAR:    %[[BITCAST_6:.*]] = llvm.bitcast %[[VAL_6]] : i16 to bf16
// CHECK-SCALAR:    %[[VAL_7:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_3]]) : (f32) -> i16
// CHECK-SCALAR:    %[[BITCAST_7:.*]] = llvm.bitcast %[[VAL_7]] : i16 to bf16
// CHECK-SCALAR:    %[[VAL_8:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_4]]) : (f32) -> i16
// CHECK-SCALAR:    %[[BITCAST_8:.*]] = llvm.bitcast %[[VAL_8]] : i16 to bf16
// CHECK-SCALAR:    %[[VAL_9:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[VAL_5]]) : (f32) -> i16
// CHECK-SCALAR:    %[[BITCAST_9:.*]] = llvm.bitcast %[[VAL_9]] : i16 to bf16
// CHECK-SCALAR:    %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_11:.*]] = llvm.insertvalue %[[BITCAST_6]], %[[VAL_10]][0] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_12:.*]] = llvm.insertvalue %[[BITCAST_7]], %[[VAL_11]][1] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_13:.*]] = llvm.insertvalue %[[BITCAST_8]], %[[VAL_12]][2] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_14:.*]] = llvm.insertvalue %[[BITCAST_9]], %[[VAL_13]][3] : !llvm.struct<(bf16, bf16, bf16, bf16)>

    %1 = arith.truncf %arg0 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
// CHECK-SCALAR:    llvm.return %[[VAL_14]] : !llvm.struct<(bf16, bf16, bf16, bf16)>

    tt.return %1: tensor<512xbf16, #blocked>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @bfloat_to_float_conversion(
// CHECK-SCALAR:                                             %[[VAL_0:.*]]: !llvm.struct<(bf16, bf16, bf16, bf16)>) -> !llvm.struct<(f32, f32, f32, f32)>
  tt.func @bfloat_to_float_conversion(%arg0 : tensor<512xbf16, #blocked>) ->  tensor<512xf32, #blocked>{
// CHECK-SCALAR:    %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(bf16, bf16, bf16, bf16)>
// CHECK-SCALAR:    %[[BITCAST_2:.*]] = llvm.bitcast %[[VAL_2]] : bf16 to i16
// CHECK-SCALAR:    %[[VAL_6:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[BITCAST_2]]) : (i16) -> f32
// CHECK-SCALAR:    %[[BITCAST_3:.*]] = llvm.bitcast %[[VAL_3]] : bf16 to i16
// CHECK-SCALAR:    %[[VAL_7:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[BITCAST_3]]) : (i16) -> f32
// CHECK-SCALAR:    %[[BITCAST_4:.*]] = llvm.bitcast %[[VAL_4]] : bf16 to i16
// CHECK-SCALAR:    %[[VAL_8:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[BITCAST_4]]) : (i16) -> f32
// CHECK-SCALAR:    %[[BITCAST_5:.*]] = llvm.bitcast %[[VAL_5]] : bf16 to i16
// CHECK-SCALAR:    %[[VAL_9:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%[[BITCAST_5]]) : (i16) -> f32
// CHECK-SCALAR:    %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_10]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_11]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_12]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK-SCALAR:    %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_13]][3] : !llvm.struct<(f32, f32, f32, f32)>

    %1 = arith.extf %arg0 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
// CHECK-SCALAR:    llvm.return %[[VAL_14]] : !llvm.struct<(f32, f32, f32, f32)>

    tt.return %1: tensor<512xf32, #blocked>
  }
}
