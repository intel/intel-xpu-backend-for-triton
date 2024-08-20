// RUN: env TRITON_INTEL_ADVANCED_PATH=1 \
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {"triton_intel_gpu.support_sg_2d_block", "triton_intel_gpu.support_dpas", "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_scalar(
// CHECK-SAME:                                         %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32) -> vector<4xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.poison : vector<4xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.insertelement %[[VAL_0]], %[[VAL_8]]{{\[}}%[[VAL_9]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.insertelement %[[VAL_1]], %[[VAL_10]]{{\[}}%[[VAL_11]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.insertelement %[[VAL_2]], %[[VAL_12]]{{\[}}%[[VAL_13]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.insertelement %[[VAL_3]], %[[VAL_14]]{{\[}}%[[VAL_15]] : i32] : vector<4xf32>
// CHECK:           llvm.return %[[VAL_16]] : vector<4xf32>
// CHECK:         }
  tt.func @test_scalar(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>, %arg3: tensor<1x16xf32>) -> tensor<4x16xf32> {
    %0 = triton_intel_gpu.glue %arg0, %arg1, %arg2, %arg3 : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<4x16xf32>
    tt.return %0 : tensor<4x16xf32>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @test_vec(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.poison : vector<8xf32>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_5]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.insertelement %[[VAL_7]], %[[VAL_4]]{{\[}}%[[VAL_6]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_9]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = llvm.insertelement %[[VAL_11]], %[[VAL_8]]{{\[}}%[[VAL_10]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_13]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_16:.*]] = llvm.insertelement %[[VAL_15]], %[[VAL_12]]{{\[}}%[[VAL_14]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_17]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_20:.*]] = llvm.insertelement %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_18]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_21]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_24:.*]] = llvm.insertelement %[[VAL_23]], %[[VAL_20]]{{\[}}%[[VAL_22]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_25]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_28:.*]] = llvm.insertelement %[[VAL_27]], %[[VAL_24]]{{\[}}%[[VAL_26]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_29]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_32:.*]] = llvm.insertelement %[[VAL_31]], %[[VAL_28]]{{\[}}%[[VAL_30]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_33]] : i32] : vector<4xf32>
// CHECK:           %[[VAL_36:.*]] = llvm.insertelement %[[VAL_35]], %[[VAL_32]]{{\[}}%[[VAL_34]] : i32] : vector<8xf32>
// CHECK:           llvm.return %[[VAL_36]] : vector<8xf32>
// CHECK:         }
  tt.func @test_vec(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<8x16xf32> {
    %0 = triton_intel_gpu.glue %arg0, %arg1 : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }
}
