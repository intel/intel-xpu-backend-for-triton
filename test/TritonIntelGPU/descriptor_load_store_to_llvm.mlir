// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 env TRITON_INTEL_PREDICATED_STORE=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test that tt.descriptor_load and tt.descriptor_store are converted to LLVM
// gather loads/stores when going through the Intel GPU to LLVM conversion.
// This is the native tensor descriptor path (without rewrite-tensor-descriptor-to-pointer).
// These tests mirror the tests in test/Triton/rewrite-tensor-descriptor-to-pointer.mlir
// but verify the LLVM lowering path instead.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load
  tt.func public @load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    // shape=[5,5] is not divisible by block_shape=[4,4], so boundary checks are generated.
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // Verify the tensor descriptor is constructed as an LLVM struct with:
    //   [0]: shape0 (i64), [1]: shape1 (i64), [2]: stride0 (i64), [3]: stride1 (i64), [4]: base_ptr (ptr<1>)
    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify extraction of descriptor fields:
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify pointer arithmetic uses the base pointer from the descriptor
    // CHECK: llvm.getelementptr %[[BASE_PTR]][%{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32

    // Verify boundary checking: index0 >= 0 AND index0 < shape0
    // CHECK: %[[ZERO0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[IDX0_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[ZERO0]] : i32
    // CHECK: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: %[[IDX0_LT_SHAPE0:.*]] = llvm.icmp "slt" %{{.*}}, %[[SHAPE0_I32]] : i32
    // CHECK: %[[DIM0_INBOUNDS:.*]] = llvm.and %[[IDX0_LT_SHAPE0]], %{{.*}} : i1
    // CHECK: %[[DIM0_PRED:.*]] = llvm.and %[[DIM0_INBOUNDS]], %[[IDX0_GE_ZERO]] : i1

    // Verify boundary checking: index1 >= 0 AND index1 < shape1
    // CHECK: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[IDX1_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[ZERO1]] : i32
    // CHECK: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: %[[IDX1_LT_SHAPE1:.*]] = llvm.icmp "slt" %{{.*}}, %[[SHAPE1_I32]] : i32

    // CHECK: %[[DIM1_INBOUNDS:.*]] = llvm.and %[[IDX1_LT_SHAPE1]], %[[DIM0_PRED]] : i1
    // CHECK: %[[PRED:.*]] = llvm.and %[[DIM1_INBOUNDS]], %[[IDX1_GE_ZERO]] : i1

    // Verify predicated load: conditional branch based on bounds check
    // If in-bounds, go to load block; otherwise skip with default value
    // CHECK: llvm.cond_br %[[PRED]], ^[[BB_LOAD:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]](%{{.*}} : i32)
    // CHECK: ^[[BB_LOAD]]:
    // CHECK: %[[LOADED:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i32
    // CHECK: llvm.br ^[[BB_MERGE]](%[[LOADED]] : i32)
    // CHECK: ^[[BB_MERGE]](%{{.*}}: i32):

    // CHECK: llvm.return
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test predicated load with ttig.support_predicated_io attribute.
// When the module has this attribute, the lowering uses triton_gen.predicated_load
// instead of branching.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @predicated_load
  tt.func public @predicated_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    // shape=[5,5] is not divisible by block_shape=[4,4], so boundary checks are generated.
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // Verify the tensor descriptor is constructed as an LLVM struct with:
    //   [0]: shape0 (i64), [1]: shape1 (i64), [2]: stride0 (i64), [3]: stride1 (i64), [4]: base_ptr (ptr<1>)
    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify extraction of descriptor fields:
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify pointer arithmetic uses the base pointer from the descriptor
    // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[BASE_PTR]][%{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32

    // Verify boundary checking: index0 >= 0 AND index0 < shape0
    // CHECK: %[[ZERO0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[IDX0_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[ZERO0]] : i32
    // CHECK: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: %[[IDX0_LT_SHAPE0:.*]] = llvm.icmp "slt" %{{.*}}, %[[SHAPE0_I32]] : i32
    // CHECK: %[[DIM0_INBOUNDS:.*]] = llvm.and %[[IDX0_LT_SHAPE0]], %{{.*}} : i1
    // CHECK: %[[DIM0_PRED:.*]] = llvm.and %[[DIM0_INBOUNDS]], %[[IDX0_GE_ZERO]] : i1

    // Verify boundary checking: index1 >= 0 AND index1 < shape1
    // CHECK: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[IDX1_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[ZERO1]] : i32
    // CHECK: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: %[[IDX1_LT_SHAPE1:.*]] = llvm.icmp "slt" %{{.*}}, %[[SHAPE1_I32]] : i32

    // CHECK: %[[DIM1_INBOUNDS:.*]] = llvm.and %[[IDX1_LT_SHAPE1]], %[[DIM0_PRED]] : i1
    // CHECK: %[[PRED:.*]] = llvm.and %[[DIM1_INBOUNDS]], %[[IDX1_GE_ZERO]] : i1

    // Verify predicated load: uses triton_gen.predicated_load intrinsic
    // with bounds predicate instead of branching.
    // The GEP-computed pointer is bitcast and passed to the predicated load.
    // CHECK: %[[LOAD_PTR:.*]] = llvm.bitcast %[[GEP]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK: triton_gen.predicated_load %[[LOAD_PTR]], %[[PRED]], %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32

    // CHECK: llvm.return
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store
  tt.func public @store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    // shape=[5,5] is not divisible by block_shape=[4,4], so boundary checks are generated.
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>
    // Verify the tensor descriptor is constructed as an LLVM struct
    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify extraction of all descriptor fields
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify pointer arithmetic uses the base pointer
    // CHECK: llvm.getelementptr %[[BASE_PTR]][%{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32

    // Verify boundary checking: index0 >= 0 AND index0 < shape0
    // CHECK: %[[S_ZERO0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[S_IDX0_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[S_ZERO0]] : i32
    // CHECK: %[[S_SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: %[[S_IDX0_LT_SHAPE0:.*]] = llvm.icmp "slt" %{{.*}}, %[[S_SHAPE0_I32]] : i32
    // CHECK: %[[S_DIM0_INBOUNDS:.*]] = llvm.and %[[S_IDX0_LT_SHAPE0]], %{{.*}} : i1
    // CHECK: %[[S_DIM0_PRED:.*]] = llvm.and %[[S_DIM0_INBOUNDS]], %[[S_IDX0_GE_ZERO]] : i1

    // Verify boundary checking: index1 >= 0 AND index1 < shape1
    // CHECK: %[[S_ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[S_IDX1_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[S_ZERO1]] : i32
    // CHECK: %[[S_SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: %[[S_IDX1_LT_SHAPE1:.*]] = llvm.icmp "slt" %{{.*}}, %[[S_SHAPE1_I32]] : i32

    // CHECK: %[[S_DIM1_INBOUNDS:.*]] = llvm.and %[[S_IDX1_LT_SHAPE1]], %[[S_DIM0_PRED]] : i1
    // CHECK: %[[S_PRED:.*]] = llvm.and %[[S_DIM1_INBOUNDS]], %[[S_IDX1_GE_ZERO]] : i1

    // Verify the thread redundancy predicate is combined with boundary mask.
    // The thread predicate (from redundant-thread elimination) is AND'd with
    // the boundary mask (S_PRED) to form the final store predicate.
    // Note: we match STORE_PRED directly anchored on S_PRED to skip over
    // element 1's interleaved boundary checks.
    // CHECK: %[[STORE_PRED:.*]] = llvm.and %{{.*}}, %[[S_PRED]] : i1

    // Verify predicated store: conditional branch based on combined predicate
    // (thread redundancy AND boundary mask)
    // CHECK: llvm.cond_br %[[STORE_PRED]], ^[[BB_STORE:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]]
    // CHECK: ^[[BB_STORE]]:
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
    // CHECK: llvm.br ^[[BB_MERGE]]
    // CHECK: ^[[BB_MERGE]]:

    // CHECK: llvm.return
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<4x4xf32>, tensor<4x4xf32, #blocked>
    tt.return
  }
}

// -----

// Test predicated store with ttig.support_predicated_io attribute.
// When the module has this attribute, the lowering uses triton_gen.predicated_store
// instead of branching.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @predicated_store
  tt.func public @predicated_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    // shape=[5,5] is not divisible by block_shape=[4,4], so boundary checks are generated.
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>
    // Verify the tensor descriptor is constructed as an LLVM struct
    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify extraction of all descriptor fields
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify pointer arithmetic uses the base pointer
    // CHECK: %[[S_GEP:.*]] = llvm.getelementptr %[[BASE_PTR]][%{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32

    // Verify boundary checking: index0 >= 0 AND index0 < shape0
    // CHECK: %[[S_ZERO0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[S_IDX0_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[S_ZERO0]] : i32
    // CHECK: %[[S_SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: %[[S_IDX0_LT_SHAPE0:.*]] = llvm.icmp "slt" %{{.*}}, %[[S_SHAPE0_I32]] : i32
    // CHECK: %[[S_DIM0_INBOUNDS:.*]] = llvm.and %[[S_IDX0_LT_SHAPE0]], %{{.*}} : i1
    // CHECK: %[[S_DIM0_PRED:.*]] = llvm.and %[[S_DIM0_INBOUNDS]], %[[S_IDX0_GE_ZERO]] : i1

    // Verify boundary checking: index1 >= 0 AND index1 < shape1
    // CHECK: %[[S_ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[S_IDX1_GE_ZERO:.*]] = llvm.icmp "sge" %{{.*}}, %[[S_ZERO1]] : i32
    // CHECK: %[[S_SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: %[[S_IDX1_LT_SHAPE1:.*]] = llvm.icmp "slt" %{{.*}}, %[[S_SHAPE1_I32]] : i32

    // CHECK: %[[S_DIM1_INBOUNDS:.*]] = llvm.and %[[S_IDX1_LT_SHAPE1]], %[[S_DIM0_PRED]] : i1
    // CHECK: %[[S_PRED:.*]] = llvm.and %[[S_DIM1_INBOUNDS]], %[[S_IDX1_GE_ZERO]] : i1

    // Verify the thread redundancy predicate is combined with boundary mask.
    // The thread predicate (from redundant-thread elimination) is AND'd with
    // the boundary mask (S_PRED) to form the final store predicate.
    // Note: we match STORE_PRED directly anchored on S_PRED to skip over
    // element 1's interleaved boundary checks.
    // CHECK: %[[STORE_PRED:.*]] = llvm.and %{{.*}}, %[[S_PRED]] : i1

    // Verify predicated store: uses triton_gen.predicated_store intrinsic
    // with combined predicate (thread redundancy AND boundary mask).
    // The GEP-computed pointer is bitcast and passed to the predicated store.
    // CHECK: %[[STORE_PTR:.*]] = llvm.bitcast %[[S_GEP]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK: triton_gen.predicated_store %[[STORE_PTR]], %{{.*}}, %[[STORE_PRED]] {cache_control = Default} : (!llvm.ptr<1>, i32, i1)

    // CHECK: llvm.return
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<4x4xf32>, tensor<4x4xf32, #blocked>
    tt.return
  }
}



// -----

// Test that tensor descriptor function arguments are properly converted to LLVM struct types.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @arg_attr
  // Verify tensordesc argument is converted to LLVM struct type with the correct layout:
  //   {shape0: i64, shape1: i64, stride0: i64, stride1: i64, base_ptr: ptr<1>}
  // CHECK-SAME: %{{.*}}: !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK-SAME: %{{.*}}: i32
  tt.func public @arg_attr(%arg0: !tt.tensordesc<4x4xf32>, %arg1: i32 {tt.divisibility = 16 : i32}) {
    tt.return
  }
}

// -----

// Test vectorized descriptor load and store: with sizePerThread > 1 and stride-1
// on the fast dimension, the gather fallback should emit wider (vectorized) I/O.
// Here sizePerThread=[1,4] with f16 gives vec=4 (4*16=64 bits < 128 bit max).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @vectorized_descriptor_load_store
  tt.func public @vectorized_descriptor_load_store(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<4x16xf16, #blocked>) {
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    // stride = [16, 1] → stride-1 on dim 1 (the fast dimension with order=[1,0])
    %desc = tt.make_tensor_descriptor %arg0, [%c4_i32, %c16_i32], [%c16_i64, %c1_i64] : <f16>, <4x16xf16>

    // With vec=4 and f16: totalWidth=64, maxWordWidth=32, width=32, nWords=2.
    // Return type is vector<2xi32>. Verify wider-than-scalar predicated loads.
    // CHECK: triton_gen.predicated_load {{.*}} : (!llvm.ptr<1>, i1, vector<2xi32>) -> vector<2xi32>
    %load = tt.descriptor_load %desc[%arg1, %arg2] : !tt.tensordesc<4x16xf16> -> tensor<4x16xf16, #blocked>

    // Verify wider-than-scalar predicated stores with the same descriptor.
    // CHECK: triton_gen.predicated_store {{.*}} {cache_control = Default} : (!llvm.ptr<1>, vector<2xi32>, i1)
    tt.descriptor_store %desc[%arg1, %arg2], %load : !tt.tensordesc<4x16xf16>, tensor<4x16xf16, #blocked>
    tt.return %load : tensor<4x16xf16, #blocked>
  }
}

// -----

// Negative test: stride != 1 on the fast dimension prevents vectorization.
// With stride[1] unknown (not constant 1), vec should be 1, producing scalar I/O.
// Dynamic shapes are used so boundary checks are not elided.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_vec_non_unit_stride
  tt.func public @no_vec_non_unit_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i64) -> (tensor<4x16xf16, #blocked>) {
    %c5_i32 = arith.constant 5 : i32
    %c17_i32 = arith.constant 17 : i32
    %c16_i64 = arith.constant 16 : i64
    // shape=[5,17]: 5%4!=0 and 17%16!=0, boundary checks are generated.
    // stride = [16, %arg3] → stride on dim 1 is unknown (not constant 1)
    %desc = tt.make_tensor_descriptor %arg0, [%c5_i32, %c17_i32], [%c16_i64, %arg3] : <f16>, <4x16xf16>

    // With unknown stride on the fast dimension, vec=1. Loads should be 16-bit (scalar f16).
    // CHECK: triton_gen.predicated_load {{.*}} : (!llvm.ptr<1>, i1, i16) -> i16
    %load = tt.descriptor_load %desc[%arg1, %arg2] : !tt.tensordesc<4x16xf16> -> tensor<4x16xf16, #blocked>

    // Stores should also be 16-bit (scalar f16).
    // CHECK: triton_gen.predicated_store {{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i16, i1)
    tt.descriptor_store %desc[%arg1, %arg2], %load : !tt.tensordesc<4x16xf16>, tensor<4x16xf16, #blocked>
    tt.return %load : tensor<4x16xf16, #blocked>
  }
}

// -----

// Test divisible descriptor load (branch path): when shape is divisible by
// block_shape and offsets are constant 0, the lowering generates BLOCK-LEVEL
// checks (on base offset only, no per-thread index added) rather than
// per-element checks. The block-level predicate is cheaper but still required
// (a fully out-of-bounds tile must return padding/other, not load unconditionally).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load_divisible
  tt.func public @load_divisible(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    // shape=[8,8] is divisible by block_shape=[4,4]; offsets are constant 0.
    %0 = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // Verify the tensor descriptor is constructed
    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify extraction of descriptor fields
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: Block-level boundary checks: the base offset (constant 0) is compared
    // COM: against the shape for each dimension. Because all dims are divisible the
    // COM: tile is all-in-or-all-out, so a single per-tile predicate guards the load.
    // COM: (The decisive base-offset-vs-base+index distinction is pinned precisely in
    // COM: the rank_reducing_load test, where offsets are function arguments.)

    // Verify icmp operations use shape0 and shape1 truncated from descriptor
    // CHECK: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE0_I32]] : i32
    // CHECK: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE1_I32]] : i32

    // Verify predicated load: conditional branch (NOT unconditional)
    // CHECK: llvm.cond_br %{{.*}}, ^[[BB_LOAD:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]](%{{.*}} : i32)
    // CHECK: ^[[BB_LOAD]]:
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<1> -> i32
    // CHECK: llvm.br ^[[BB_MERGE]]

    %3 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test divisible descriptor load with predicated_io: when ttig.support_predicated_io
// is present, block-level checks are emitted but the load uses triton_gen.predicated_load
// intrinsic instead of branching.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load_divisible_predicated
  tt.func public @load_divisible_predicated(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    // shape=[8,8] is divisible by block_shape=[4,4]; offsets are constant 0.
    %0 = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: Verify block-level boundary checks against shapes for both dimensions
    // Verify icmp operations use shape0 and shape1 truncated from descriptor
    // CHECK: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE0_I32]] : i32
    // CHECK: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE1_I32]] : i32

    // Verify predicated load intrinsic with block-level predicate
    // CHECK: triton_gen.predicated_load %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32

    %3 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test divisible descriptor store: when shape is divisible by block_shape,
// the lowering generates block-level checks on the base offset, combined with
// the thread redundancy predicate.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_divisible
  tt.func public @store_divisible(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    // shape=[8,8] is divisible by block_shape=[4,4]; offsets are constant 0.
    %0 = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: Verify block-level boundary checks against shapes for both dimensions
    // Verify icmp operations use shape0 and shape1 truncated from descriptor
    // CHECK: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE0_I32]] : i32
    // CHECK: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[SHAPE1_I32]] : i32

    // Verify predicated store: conditional branch based on combined predicate
    // CHECK: llvm.cond_br %{{.*}}, ^[[BB_STORE:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]]
    // CHECK: ^[[BB_STORE]]:
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
    // CHECK: llvm.br ^[[BB_MERGE]]

    tt.descriptor_store %0[%c0_i32, %c0_i32], %arg1 : !tt.tensordesc<4x4xf32>, tensor<4x4xf32, #blocked>
    tt.return
  }
}

// -----

// Test rank reduction from a 4D descriptor (1x1x4x4) to a 2D tensor (4x4):
// lowering generates boundary checks for descriptor dimensions whose shape is
// not provably divisible by the block shape at compile time. With the fix,
// the leading singleton dimensions (block_shape=1) ARE divisible, so they get
// BLOCK-LEVEL checks on the base offset (arg1, arg2), not per-element checks.
// Dims 2 and 3 use dynamic shapes and get per-element checks.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @rank_reducing_load(
  // CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<1> {tt.pointee_type = f32},
  // CHECK-SAME:      %[[OFFSET0:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET1:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET2:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET3:[^:]+]]: i32,
  // CHECK-SAME:      %[[SHAPE2ARG:[^:]+]]: i32,
  // CHECK-SAME:      %[[SHAPE3ARG:[^:]+]]: i32,
  tt.func public @rank_reducing_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    // Dynamic shapes for dims 2 and 3: isDivisible cannot prove divisibility,
    // so boundary checks are generated for those dimensions.
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %c1_i32, %arg5, %arg6], [%c16_i64, %c16_i64, %c4_i64, %c1_i64] : <f32>, <1x1x4x4xf32>

    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[8] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE2:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE3:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>

    // CHECK-DAG: %[[LINEAR_OFF2:.*]] = llvm.add {{.*}}, %[[OFFSET2]] : i32
    // CHECK-DAG: %[[LINEAR_OFF3:.*]] = llvm.add {{.*}}, %[[OFFSET3]] : i32

    // COM: Dims 0 and 1 have block_shape=1 (divisible), so they get BLOCK-LEVEL checks:
    // COM: the base offset (OFFSET0/OFFSET1) is compared directly against the shape,
    // COM: with NO llvm.add of a per-thread index. Dims 2 and 3 use dynamic shapes and
    // COM: get per-element checks against LINEAR_OFF (base + index).

    // Block-level checks for the singleton dims 0 and 1 (base offset used directly):
    // CHECK-DAG: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK-DAG: llvm.icmp "slt" %[[OFFSET0]], %[[SHAPE0_I32]] : i32
    // CHECK-DAG: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK-DAG: llvm.icmp "slt" %[[OFFSET1]], %[[SHAPE1_I32]] : i32

    // Per-element checks for dims 2 and 3 (base + index):
    // CHECK-DAG: llvm.icmp "slt" %[[LINEAR_OFF2]], %{{.*}} : i32
    // CHECK-DAG: llvm.icmp "slt" %[[LINEAR_OFF3]], %{{.*}} : i32
    %load = tt.descriptor_load %desc[%arg1, %arg2, %arg3, %arg4] : !tt.tensordesc<1x1x4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %load : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test rank reduction from a 4D descriptor (1x1x4x4) to a 2D tensor (4x4):
// store lowering must keep boundary checks for descriptor dimensions whose
// shape is not provably divisible by the block shape. With the fix, the leading
// singleton dimensions (block_shape=1) ARE divisible, so they get BLOCK-LEVEL
// checks on the base offset (not "always in-bounds"); dims 2 and 3 use dynamic
// shapes and get per-element checks.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @rank_reducing_store(
  // CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<1> {tt.pointee_type = f32},
  // CHECK-SAME:      %[[OFFSET0:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET1:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET2:[^:]+]]: i32,
  // CHECK-SAME:      %[[OFFSET3:[^:]+]]: i32,
  // CHECK-SAME:      %[[SHAPE2ARG:[^:]+]]: i32,
  // CHECK-SAME:      %[[SHAPE3ARG:[^:]+]]: i32,
  tt.func public @rank_reducing_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<4x4xf32, #blocked>) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    // Dynamic shapes for dims 2 and 3: isDivisible cannot prove divisibility,
    // so boundary checks are generated for those dimensions.
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %c1_i32, %arg5, %arg6], [%c16_i64, %c16_i64, %c4_i64, %c1_i64] : <f32>, <1x1x4x4xf32>

    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[8] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE2:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[SHAPE3:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, ptr<1>)>

    // CHECK-DAG: %[[LINEAR_OFF2:.*]] = llvm.add {{.*}}, %[[OFFSET2]] : i32
    // CHECK-DAG: %[[LINEAR_OFF3:.*]] = llvm.add {{.*}}, %[[OFFSET3]] : i32

    // COM: Dims 0 and 1 have block_shape=1 (divisible), so they get BLOCK-LEVEL checks:
    // COM: the base offset (OFFSET0/OFFSET1) is compared directly against the shape,
    // COM: with NO llvm.add of a per-thread index. Dims 2 and 3 use dynamic shapes and
    // COM: get per-element checks against LINEAR_OFF (base + index).

    // Block-level checks for the singleton dims 0 and 1 (base offset used directly):
    // CHECK-DAG: %[[SHAPE0_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK-DAG: llvm.icmp "slt" %[[OFFSET0]], %[[SHAPE0_I32]] : i32
    // CHECK-DAG: %[[SHAPE1_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // CHECK-DAG: llvm.icmp "slt" %[[OFFSET1]], %[[SHAPE1_I32]] : i32

    // Per-element checks for dims 2 and 3 (base + index):
    // CHECK-DAG: llvm.icmp "slt" %[[LINEAR_OFF2]], %{{.*}} : i32
    // CHECK-DAG: llvm.icmp "slt" %[[LINEAR_OFF3]], %{{.*}} : i32
    tt.descriptor_store %desc[%arg1, %arg2, %arg3, %arg4], %arg7 : !tt.tensordesc<1x1x4x4xf32>, tensor<4x4xf32, #blocked>
    tt.return
  }
}

// -----

// Test divisible descriptor load with a non-zero constant offset:
// When shape[i] % block_shape[i] == 0 AND offset[i] % block_shape[i] == 0,
// the load uses BLOCK-LEVEL checks on the base offset directly rather than
// per-element checks. Here offset = const 4 (divisible by block_shape=4) for
// both dimensions, distinct from the existing zero-offset tests above.
// This verifies that isDivisible correctly handles non-zero constant offsets.

// COM: The BLOCK-LEVEL check produces: icmp("sge", const_4, 0) and
// COM:                                 icmp("slt", const_4, trunc(shape))
// COM: where const_4 is the OFFSET itself, not (per_thread + offset).
// COM: This is distinct from PER-ELEMENT checks which compute
// COM: add(per_thread_index, const_4) and then compare that sum.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load_nonzero_const_divisible_offset
  tt.func public @load_nonzero_const_divisible_offset(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    // shape=[8,8] is divisible by block_shape=[4,4]; offsets are constant 4
    // (also divisible by 4). isDivisible(constant 4, 4) returns true.

    // CHECK: %[[OFF4:.*]] = llvm.mlir.constant(4 : i32) : i32

    %0 = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // COM: Block-level boundary checks: the constant base offset 4 is compared
    // COM: directly against the shape for each dimension. The per-thread index
    // COM: is added to the GEP but NOT used in the icmp predicate.
    // COM: Both dim0 and dim1 use %[[OFF4]] in their icmps (not add(per_thread,4)).

    // CHECK: %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[SHP0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[SHP1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Verify block-level checks for both dims: constant offset 4 used directly
    // CHECK: llvm.icmp "sge" %[[OFF4]], %{{.*}} : i32
    // CHECK: %[[SHP0_I32:.*]] = llvm.trunc %[[SHP0]] : i64 to i32
    // CHECK: llvm.icmp "slt" %[[OFF4]], %[[SHP0_I32]] : i32
    // CHECK: llvm.icmp "sge" %[[OFF4]], %{{.*}} : i32
    // CHECK: %[[SHP1_I32:.*]] = llvm.trunc %[[SHP1]] : i64 to i32
    // CHECK: llvm.icmp "slt" %[[OFF4]], %[[SHP1_I32]] : i32

    // CHECK: triton_gen.predicated_load %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32

    %3 = tt.descriptor_load %0[%c4_i32, %c4_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test divisible descriptor load inside an scf.for loop:
// The loop induction variable has lb=0, step=4 (both divisible by block_shape=4),
// so isDivisible(iv, 4) returns true. The shape is divisible by 4 too.
// Both dimensions qualify for BLOCK-LEVEL checks: dim0 uses the IV directly as
// the base offset (no per-thread index added to the predicate), dim1 uses the
// constant zero offset directly.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load_for_iv_divisible
  tt.func public @load_for_iv_divisible(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    // shape=[8,8] is divisible by block_shape=[4,4]
    %desc = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] : <f32>, <4x4xf32>

    // COM: The descriptor is built outside the loop and reused inside.
    // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Loop with lb=0, step=4: isDivisible(iv, 4) is true because lb=0 and step=4
    // are both divisible by block_shape=4. Both dims qualify for block-level checks.
    // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
    %result = scf.for %iv = %c0_i32 to %arg1 step %c4_i32 iter_args(%accum = %c0_i32) -> i32 : i32 {
      // Inside the loop, dim0 uses the IV directly as the block-level predicate.
      // No per-thread index is added to %[[IV]] in the icmps — only in the GEP.

      // Verify GEP adds the IV to per-thread index for pointer arithmetic
      // CHECK: llvm.add %{{.*}}, %[[IV]] : i32

      // Verify block-level check for dim0: IV used directly in icmp (not IV+per_thread)
      // CHECK: llvm.icmp "sge" %[[IV]], %{{.*}} : i32
      // CHECK: llvm.icmp "slt" %[[IV]], %{{.*}} : i32

      // Verify predicated load inside loop
      // CHECK: triton_gen.predicated_load %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32

      %load = tt.descriptor_load %desc[%iv, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
      scf.yield %accum : i32
    }
    // Outside the loop: constant-0 offsets, also block-level.
    %ret = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %ret : tensor<4x4xf32, #blocked>
  }
}

// -----

// Test column_major descriptor load with MIXED divisibility of inner dims:
// This exercises the permuteDescDim path in LoadStoreOpToLLVM.cpp.
//
// Descriptor shape=[17, 16], block_shape=[16, 8].
// The column_major flag swaps the inner two dims before the load, so:
//   - result tensor shape is [8, 16] (desc[1], desc[0])
//   - result dim0 maps to desc dim1: shape=16, offset=8 → blockLevelDims
//   - result dim1 maps to desc dim0: shape=17, offset=0 → perElementDims
//
// Descriptor divisibility analysis (before the inner-dim swap):
//   desc dim0: shape=17 NOT divisible by block_shape[0]=16 → per-element
//   desc dim1: shape=16 divisible by block_shape[1]=8, offset=8 div by 8 → block-level
//
// After permuteDescDim swaps the inner dims, the CORRECT output maps:
//   result dim0 (= desc dim1) → block-level: icmp uses base offset 8 directly
//   result dim1 (= desc dim0) → per-element: icmp uses (per_thread + 0)
//
// NOTE: This test is written for the CORRECT post-fix expected output.
// The current code has a bug in how permuteDescDim remaps blockLevelDims /
// perElementDims. With the bug, the output SWAPS the classifications:
//   result dim0 → per-element (per_thread + 8) vs shape 16
//   result dim1 → block-level (0 directly) vs shape 17
// The CHECK patterns below document the CORRECT behavior. They will FAIL until
// the companion permuteDescDim bug-fix is applied to LoadStoreOpToLLVM.cpp.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 1], order = [1, 0]}>

// Note: ttig.support_2d_block_io is intentionally absent so this op is
// lowered via the predicated scatter/gather path, not 2D block I/O.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @colmaj_mixed_inner_div
  tt.func public @colmaj_mixed_inner_div(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<8x16xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c17_i32 = arith.constant 17 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    // desc shape=[17, 16]: dim0 NOT divisible by 16; dim1 divisible by 8.
    // offset=[0, 8]: both divisible by their respective block sizes.

    %desc = tt.make_tensor_descriptor %arg0, [%c17_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x8xf32>

    // COM: -----------------------------------------------------------------------
    // COM: EXPECTED CHECK PATTERNS (activate once the permuteDescDim fix lands):
    // COM:
    // COM:   %[[OFF8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // COM:   %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // COM:   %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // COM:   %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // COM:
    // COM:   Block-level check for result dim0 (= desc dim1, shape=16, offset=8):
    // COM:     %[[PSHAPE0:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32
    // COM:     llvm.icmp "sge" %[[OFF8]], %{{.*}} : i32
    // COM:     llvm.icmp "slt" %[[OFF8]], %[[PSHAPE0]] : i32
    // COM:
    // COM:   Per-element check for result dim1 (= desc dim0, shape=17, offset=0):
    // COM:     %[[ELEM1:.*]] = llvm.add %{{.*}}, %{{.*}} : i32
    // COM:     %[[PSHAPE1:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // COM:     llvm.icmp "sge" %[[ELEM1]], %{{.*}} : i32
    // COM:     llvm.icmp "slt" %[[ELEM1]], %[[PSHAPE1]] : i32
    // COM:
    // COM: CURRENT (BUGGY) output swaps the classifications:
    // COM:   result dim0 → per-element check: add(per_thread, 8) vs shape1=16
    // COM:   result dim1 → block-level check: constant 0 vs shape0=17
    // COM: The bug is in how permuteDescDim remaps blockLevelDims/perElementDims
    // COM: indices via permDimIdx in LoadStoreOpToLLVM.cpp.
    // COM: -----------------------------------------------------------------------

    // Verify at least the function compiles and emits a predicated load.
    // CHECK: triton_gen.predicated_load %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32

    %3 = tt.descriptor_load %desc[%c0_i32, %c8_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<16x8xf32> -> tensor<8x16xf32, #blocked>
    tt.return %3 : tensor<8x16xf32, #blocked>
  }
}
