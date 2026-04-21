// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// COM: Tests reduction when threads_per_warp < num_warps.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [64], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_problem_size_64_threads_per_warp_32
  tt.func @reduce_problem_size_64_threads_per_warp_32(%f : tensor<2048xi32, #blocked>) -> i32 {

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
    tt.return %g : i32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 32 : i32} {
  // CHECK-LABEL: test_reduce
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

// -----

// COM: Tests 1D integer add reduction with threads_per_warp=16 and num_warps=32.
// COM: This exercises the common Intel GPU DPAS subgroup size (16) with multi-round
// COM: inter-warp reduction: 32 warps require 2 rounds (16-wide reduce + clustered).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [32], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_tpw16_addi
  tt.func @reduce_tpw16_addi(%f : tensor<512xi32, #blocked>) -> i32 {

  // 1st round: intra-warp full reduce across 16 lanes (3 args = Reduce)
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 2nd round: inter-warp reduce with problem size 32 using 16-wide subgroups (3 args = full Reduce)
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 3rd round: inter-warp reduce with remaining 2 values using ClusteredReduce (4 args)
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiijj(%{{.*}}) {{.*}} : (i32, i32, i32, i32) -> i32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<512xi32, #blocked>) -> i32
    tt.return %g : i32
  }
}

// -----

// COM: Tests 1D integer max reduction with threads_per_warp=16.
// COM: Verifies that non-add operations (arith.maxsi) also use SPIR-V group
// COM: reduce intrinsics (GroupNonUniformSMax) with 16-wide subgroups.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_tpw16_maxsi
  tt.func @reduce_tpw16_maxsi(%f : tensor<64xi32, #blocked>) -> i32 {

  // 1st round: intra-warp full reduce across 16 lanes using GroupNonUniformSMax (3 args = Reduce)
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformSMaxiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

  // 2nd round: inter-warp reduce with 4 warps using ClusteredReduce (4 args)
  // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformSMaxiijj(%{{.*}}) {{.*}} : (i32, i32, i32, i32) -> i32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %max = arith.maxsi %arg0, %arg1 : i32
      tt.reduce.return %max : i32
    }) {axis = 0 : i32} : (tensor<64xi32, #blocked>) -> i32
    tt.return %g : i32
  }
}

// -----

// COM: Tests 2D float add reduction along axis=0 with threads_per_warp=16.
// COM: Threads are distributed along dim 1, warps along dim 0. The intra-warp
// COM: reduction is a no-op (no lanes along axis 0), so this exercises the
// COM: inter-warp shared memory reduction path with 16-wide subgroups.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_tpw16_2d_axis0
  tt.func @reduce_tpw16_2d_axis0(%f : tensor<4x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {

  // COM: No intra-warp reduce (threads are [1, 16], no lanes along dim 0).
  // COM: Inter-warp: 4 warps store to shared memory, then ClusteredReduce.

  // Store partial results to shared memory (may be vectorized or scalarized)
  // CHECK:     llvm.store %{{.*}}, %{{.*}} : {{.*}}, !llvm.ptr<3>
  // CHECK:     llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {{.*}} : (i32) -> ()

  // Inter-warp reduce: load from shared memory and ClusteredReduce with cluster_size=4 (4 args)
  // CHECK:     llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}
  // CHECK:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiifj(%{{.*}}) {{.*}} : (i32, i32, f32, i32) -> f32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<4x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %g : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}
