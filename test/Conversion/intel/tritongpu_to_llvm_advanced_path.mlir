// RUN: TRITON_INTEL_ADVANCED_PATH=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#warp = #triton_intel_gpu.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK:   llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64 attributes {passthrough = ["nounwind", "willreturn", ["memory", "0"]]}
// CHECK:   llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_maxf(f32) -> f32 attributes {passthrough = ["convergent"]}
// CHECK:   llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_addf(f32) -> f32 attributes {passthrough = ["convergent"]}

// CHECK-LABEL:   llvm.func spir_kernelcc @reduce_sum(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<8xf32>) -> f32 attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
  tt.func public @reduce_sum(%arg0: tensor<8x16xf32>) -> f32 {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<8xf32> to tensor<8x16xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.extractelement %[[VAL_0]][%[[VAL_2]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_addf(%[[VAL_3]]) {{.*}} : (f32) -> f32
    %0 = triton_intel_gpu.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.addf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32>) -> f32
    tt.return %1: f32
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @reduce_max(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<8xf32>) -> f32 attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
  tt.func public @reduce_max(%arg0: tensor<8x16xf32>) -> f32 {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<8xf32> to tensor<8x16xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.extractelement %[[VAL_0]][%[[VAL_2]] : i32] : vector<8xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_maxf(%[[VAL_3]]) {{.*}} : (f32) -> f32
    %0 = triton_intel_gpu.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.maxnumf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32>) -> f32
    tt.return %1: f32
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @broadcast(
// CHECK-SAME:                                       %[[VAL_0:.*]]: f32) -> vector<16xf32>
  tt.func public @broadcast(%arg0: f32) -> tensor<16x16xf32> {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.poison : vector<1xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.insertelement %[[VAL_0]], %[[VAL_1]][%[[VAL_2]] : i32] : vector<1xf32>
// CHECK:           %[[VAL_4:.*]] = llvm.shufflevector %[[VAL_3]], %[[VAL_1]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<1xf32>
    %0 = tt.splat %arg0 : f32 -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #warp}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
    %2 = tt.broadcast %1 : tensor<16x1xf32, #warp> -> tensor<16x16xf32>
    tt.return %2 : tensor<16x16xf32>
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @addptr(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr<1>) -> !llvm.ptr<1> attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
  tt.func public @addptr(%arg0: !tt.ptr<f16>) -> !tt.ptr<f16> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.ptr<1> to !tt.ptr<f16>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.call spir_funccc @_Z12get_group_idj(%[[VAL_2]]) {{.*}} : (i32) -> i64
// CHECK:           %[[VAL_4:.*]] = llvm.trunc %[[VAL_3]] : i64 to i32
// CHECK:           %[[VAL_5:.*]] = llvm.sext %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]][%[[VAL_5]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i64
    tt.return %2 : !tt.ptr<f16>
  }
}
