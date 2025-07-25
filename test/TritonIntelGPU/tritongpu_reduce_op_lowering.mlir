// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// COM: Tests reduction when threads_per_warp < num_warps.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [64], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_problem_size_64_threads_per_warp_32
  tt.func @reduce_problem_size_64_threads_per_warp_32(%f : tensor<2048xi32, #blocked>) {

  // 1st round intra-warp reduce
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 2nd round inter-warp reduce with problem size 64 with threads_per_warp 32
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: [[PARTIAL_REDUCE_0:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 3rd round inter-warp reduce with problem size 2 with threads_per_warp 32
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: [[PARTIAL_REDUCE_1:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiijj(%{{.*}}) {{.*}} : (i32, i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // get final result
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: [[FINAL_RESULT:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<2048xi32, #blocked>) -> i32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  tt.func public @test_reduce(%arg0: tensor<32x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {
    // CHECK:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif
    // CHECK:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif
    // CHECK:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif
    // CHECK:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif
    // CHECK:     llvm.store
    // CHECK-NOT: llvm.load
    // CHECK:     llvm.call spir_funccc @_Z7barrierj
    %1 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %2 : f32
    }) {allocation.offset = 0 : i32} : (tensor<32x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}
