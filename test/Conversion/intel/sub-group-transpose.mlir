// RUN: env TRITON_INTEL_ENABLE_BLOCK_PTR=1 \
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm \
// RUN: | FileCheck %s

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr<3>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: vector<16xf32>) -> vector<16xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z16get_sub_group_idv() {{{.*}}} : () -> i32
// CHECK:           %[[VAL_5:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_idv() {{{.*}}} : () -> i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(256 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_7]], %[[VAL_4]]  : i32
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           llvm.call spir_funccc @llvm.genx.GenISA.simdBlockWrite(%[[VAL_9]], %[[VAL_1]]) {{{.*}}} : (!llvm.ptr<3>, vector<16xf32>) -> ()
// CHECK:           %[[VAL_10:.*]] = llvm.mul %[[VAL_6]], %[[VAL_5]]  : i32
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_10]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<3> -> vector<16xf32>
// CHECK:           llvm.return %[[VAL_12]] : vector<16xf32>
// CHECK:         }
  tt.func @test(%arg0: !tt.ptr<f32, 3>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = triton_intel_gpu.sub_group_transpose %arg0, %arg1 : tensor<16x16xf32>
    tt.return %0 : tensor<16x16xf32>
  }
}
