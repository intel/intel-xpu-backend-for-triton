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
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c4_i32, %c4_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <tensor<4x4xf32>>

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
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<tensor<4x4xf32>> -> tensor<4x4xf32, #blocked>
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
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c4_i32, %c4_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <tensor<4x4xf32>>

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
    // CHECK: triton_gen.predicated_load %[[LOAD_PTR]], %{{.*}}, %[[PRED]], %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i64, i1, i32) -> i32

    // CHECK: llvm.return
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<tensor<4x4xf32>> -> tensor<4x4xf32, #blocked>
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
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c4_i32, %c4_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <tensor<4x4xf32>>
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
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<4x4xf32>>, tensor<4x4xf32, #blocked>
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
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c4_i32, %c4_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <tensor<4x4xf32>>
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
    // CHECK: triton_gen.predicated_store %[[STORE_PTR]], %{{.*}}, %{{.*}}, %[[STORE_PRED]] {cache_control = Default} : (!llvm.ptr<1>, i32, i64, i1)

    // CHECK: llvm.return
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<4x4xf32>>, tensor<4x4xf32, #blocked>
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
  tt.func public @arg_attr(%arg0: !tt.tensordesc<tensor<4x4xf32>>, %arg1: i32 {tt.divisibility = 16 : i32}) {
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
    %desc = tt.make_tensor_descriptor %arg0, [%c4_i32, %c16_i32], [%c16_i64, %c1_i64] : <f16>, <tensor<4x16xf16>>

    // With vec=4 and f16: totalWidth=64, maxWordWidth=32, width=32, nWords=2.
    // Return type is vector<2xi32>. Verify wider-than-scalar predicated loads.
    // CHECK: triton_gen.predicated_load {{.*}} : (!llvm.ptr<1>, i64, i1, vector<2xi32>) -> vector<2xi32>
    %load = tt.descriptor_load %desc[%arg1, %arg2] : !tt.tensordesc<tensor<4x16xf16>> -> tensor<4x16xf16, #blocked>

    // Verify wider-than-scalar predicated stores with the same descriptor.
    // CHECK: triton_gen.predicated_store {{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, vector<2xi32>, i64, i1)
    tt.descriptor_store %desc[%arg1, %arg2], %load : !tt.tensordesc<tensor<4x16xf16>>, tensor<4x16xf16, #blocked>
    tt.return %load : tensor<4x16xf16, #blocked>
  }
}

// -----

// Negative test: stride != 1 on the fast dimension prevents vectorization.
// With stride[1] unknown (not constant 1), vec should be 1, producing scalar I/O.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_vec_non_unit_stride
  tt.func public @no_vec_non_unit_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i64) -> (tensor<4x16xf16, #blocked>) {
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    // stride = [16, %arg3] → stride on dim 1 is unknown (not constant 1)
    %desc = tt.make_tensor_descriptor %arg0, [%c4_i32, %c16_i32], [%c16_i64, %arg3] : <f16>, <tensor<4x16xf16>>

    // With unknown stride on the fast dimension, vec=1. Loads should be 16-bit (scalar f16).
    // CHECK: triton_gen.predicated_load {{.*}} : (!llvm.ptr<1>, i64, i1, i16) -> i16
    %load = tt.descriptor_load %desc[%arg1, %arg2] : !tt.tensordesc<tensor<4x16xf16>> -> tensor<4x16xf16, #blocked>

    // Stores should also be 16-bit (scalar f16).
    // CHECK: triton_gen.predicated_store {{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i16, i64, i1)
    tt.descriptor_store %desc[%arg1, %arg2], %load : !tt.tensordesc<tensor<4x16xf16>>, tensor<4x16xf16, #blocked>
    tt.return %load : tensor<4x16xf16, #blocked>
  }
}
