// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory  --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [64], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32} {
  // CHECK-LABEL: reduce_problem_size_64_threads_per_warp_32
  tt.func @reduce_problem_size_64_threads_per_warp_32(%f : tensor<2048xi32, #blocked>) {

  // 1st round intra-warp reduce
  // CHECK: [[SHUFFLE_0_0:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_0_0]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_0_1:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_0_1]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_0_2:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_0_2]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_0_3:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_0_3]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_0_4:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_0_4]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 2nd round inter-warp reduce with problem size 64 with threads_per_warp 32
  // CHECK: llvm.call @_Z7barrierj(%{{.*}}) {passthrough = ["convergent"]} : (i32) -> ()
  // CHECK: [[PARTIAL_REDUCE_0:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: [[SHUFFLE_1_0:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_1_0]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_1_1:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_1_1]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_1_2:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_1_2]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_1_3:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_1_3]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: [[SHUFFLE_1_4:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_1_4]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 3rd round inter-warp reduce with problem size 2 with threads_per_warp 32
  // CHECK: llvm.call @_Z7barrierj(%{{.*}}) {passthrough = ["convergent"]} : (i32) -> ()
  // CHECK: [[PARTIAL_REDUCE_1:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: [[SHUFFLE_2_0:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  %{{.*}} = llvm.call @_Z21sub_group_shuffle_xorij({{.*}}, [[SHUFFLE_2_0]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // get final result
  // CHECK: llvm.call @_Z7barrierj(%{{.*}}) {passthrough = ["convergent"]} : (i32) -> ()
  // CHECK: [[FINAL_RESULT:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<2048xi32, #blocked>) -> i32
    tt.return
  }
}
