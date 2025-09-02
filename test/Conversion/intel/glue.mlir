// RUN: env TRITON_INTEL_ADVANCED_PATH=1 \
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_scalar(
// CHECK-SAME:                                         %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32, %[[PTR_1:.*]]: !llvm.ptr<1>) -> vector<4xf32>
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
    %0 = ttig.glue %arg0, %arg1, %arg2, %arg3 : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<4x16xf32>
    tt.return %0 : tensor<4x16xf32>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @test_vec_2(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[PTR_1:.*]]: !llvm.ptr<1>) -> vector<8xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.shufflevector %[[VAL_0]], %[[VAL_1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<8xf32>
// CHECK:         }
  tt.func @test_vec_2(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<8x16xf32> {
    %0 = ttig.glue %arg0, %arg1 : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @test_vec_4(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>, %[[VAL_3:.*]]: vector<4xf32>, %[[PTR_1:.*]]: !llvm.ptr<1>) -> vector<16xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.shufflevector %[[VAL_0]], %[[VAL_1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.shufflevector %[[VAL_2]], %[[VAL_3]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
// CHECK:           %[[VAL_10:.*]] = llvm.shufflevector %[[VAL_8]], %[[VAL_9]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>
// CHECK:           llvm.return %[[VAL_10]] : vector<16xf32>
// CHECK:         }
  tt.func @test_vec_4(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<4x16xf32>, %arg3: tensor<4x16xf32>) -> tensor<16x16xf32> {
    %0 = ttig.glue %arg0, %arg1, %arg2, %arg3 : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<16x16xf32>
    tt.return %0 : tensor<16x16xf32>
  }
}
