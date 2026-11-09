// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// COM: Tests reduction when threads_per_warp < num_warps.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [64], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_problem_size_64_threads_per_warp_32
  tt.func @reduce_problem_size_64_threads_per_warp_32(%f : tensor<2048xi32, #blocked>) -> i32 {

  // 1st round intra-warp reduce
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<1xi32>, !llvm.ptr<3>

  // COM: Load all 64 partials, combine register pairs, then reduce 32 lanes.
  // CHECK: triton_gen.barrier {mem_fence = Local}
  // CHECK-NOT: triton_gen.barrier
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}i32
  // CHECK-NOT: triton_gen.barrier
  // CHECK: llvm.add
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // COM: The final reduction is already in registers.
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK-NOT: triton_gen.barrier
  // CHECK: llvm.return

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
    // COM: No lanes vary along axis 0, so store before reducing across warps.
    // CHECK:     llvm.store
    // CHECK-NOT: llvm.load
    // CHECK:     triton_gen.barrier {mem_fence = Local}
    // CHECK:     llvm.load
    // COM: Preserve output lanes, reducing warp partials in registers.
    // CHECK:     llvm.fadd
    // CHECK-NOT: triton_gen.barrier
    // CHECK-NOT: llvm.store
    // CHECK:     llvm.return
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
// COM: 32 warp partials fit in two registers per lane and need just one barrier.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [32], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_tpw16_addi
  tt.func @reduce_tpw16_addi(%f : tensor<512xi32, #blocked>) -> i32 {

  // 1st round: intra-warp full reduce across 16 lanes (3 args = Reduce)
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<1xi32>, !llvm.ptr<3>

  // COM: Combine register pairs, then perform a full 16-lane reduction.
  // CHECK: triton_gen.barrier {mem_fence = Local}
  // CHECK-NOT: triton_gen.barrier
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}i32
  // CHECK: llvm.add
  // CHECK: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij(%{{.*}}) {{.*}} : (i32, i32, i32) -> i32
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK-NOT: triton_gen.barrier
  // CHECK: llvm.return

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
  // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<1xi32>, !llvm.ptr<3>

  // 2nd round: inter-warp reduce with 4 warps using ClusteredReduce (4 args)
  // CHECK: triton_gen.barrier {mem_fence = Local}
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xi32>
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
// COM: inter-warp shared memory reduction path with register-local accumulation.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_tpw16_2d_axis0
  tt.func @reduce_tpw16_2d_axis0(%f : tensor<4x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {

  // COM: No intra-warp reduce (threads are [1, 16], no lanes along dim 0).
  // COM: Inter-warp: 4 warps store to shared memory, then reduce in registers.

  // Store partial results to shared memory (may be vectorized or scalarized)
  // CHECK:     llvm.store %{{.*}}, %{{.*}} : {{.*}}, !llvm.ptr<3>
  // CHECK:     triton_gen.barrier {mem_fence = Local}

  // COM: All output lanes remain in place, so no layout restoration is needed.
  // CHECK:     llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}
  // CHECK:     llvm.fadd
  // CHECK-NOT: triton_gen.barrier
  // CHECK-NOT: llvm.store
  // CHECK:     llvm.return

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<4x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %g : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}

// -----

// COM: Mixed-width operands use separate byte offsets and register permutations.
// COM: All repetitions and operands are stored before the single barrier.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: reduce_mixed_width_reps
  tt.func @reduce_mixed_width_reps(%x: tensor<4x256xi64, #blocked>, %y: tensor<4x256xi8, #blocked>) -> (tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi8, #ttg.slice<{dim = 0, parent = #blocked}>>) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : {{.*}}i64{{.*}}, !llvm.ptr<3>
    // CHECK-NOT: triton_gen.barrier
    // CHECK-NOT: llvm.load
    // CHECK: llvm.store %{{.*}}, %{{.*}} : {{.*}}i8{{.*}}, !llvm.ptr<3>
    // CHECK: triton_gen.barrier {mem_fence = Local}
    // CHECK-NOT: triton_gen.barrier
    // CHECK-NOT: llvm.store
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}i64
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> {{.*}}i8
    // CHECK-NOT: triton_gen.barrier
    // CHECK-NOT: llvm.store
    // CHECK: llvm.return
    %a, %b = "tt.reduce"(%x, %y) ({
    ^bb0(%x0: i64, %y0: i8, %x1: i64, %y1: i8):
      %sum0 = arith.addi %x0, %x1 : i64
      %sum1 = arith.addi %y0, %y1 : i8
      tt.reduce.return %sum0, %sum1 : i64, i8
    }) {axis = 0 : i32} : (tensor<4x256xi64, #blocked>, tensor<4x256xi8, #blocked>) -> (tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi8, #ttg.slice<{dim = 0, parent = #blocked}>>)
    tt.return %a, %b : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi8, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}
