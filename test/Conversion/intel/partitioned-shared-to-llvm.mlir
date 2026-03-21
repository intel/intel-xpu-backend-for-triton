// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test lowering of operations with PartitionedSharedEncodingAttr using swizzled_shared layout.
// Swizzled layouts produce runtime-dependent partition indices, so the lowering must use a
// select chain (not a vector-of-pointers extraction) to choose the correct shared memory base.
// See https://github.com/intel/intel-xpu-backend-for-triton/issues/6154
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_swizzled = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_swizzled}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: partitioned_shared_swizzled_local_alloc
  tt.func @partitioned_shared_swizzled_local_alloc(%arg0: tensor<16x16xf16, #blocked>) {
    // Verify stores go through shared memory and partition base is selected via select chain.
    // The select and store ops interleave (one select+store per element), so we check
    // that at least one select on ptr<3> appears and that all 4 stores are emitted.
    // Ensure no vector-of-pointers extraction is used (the old buggy pattern).
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NOT: llvm.extractelement {{.*}} !llvm.ptr<3>
    // CHECK: llvm.select {{.*}} !llvm.ptr<3>
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_swizzled = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_swizzled}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: partitioned_shared_swizzled_local_load
  tt.func @partitioned_shared_swizzled_local_load() -> tensor<16x16xf16, #blocked> {
    // Verify loads use select chain for partition base selection (no vector-of-pointers).
    // The select and load ops interleave, so we check that at least one select on
    // ptr<3> appears and that all 4 loads are emitted.
    // Ensure no vector-of-pointers extraction is used (the old buggy pattern).
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NOT: llvm.extractelement {{.*}} !llvm.ptr<3>
    // CHECK: llvm.select {{.*}} !llvm.ptr<3>
    // CHECK-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{[0-9]+}}xf16>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 512 : i32, 66048 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_swizzled = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_swizzled}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: partitioned_shared_swizzled_local_store
  tt.func @partitioned_shared_swizzled_local_store(%arg0: tensor<16x16xf16, #blocked>) {
    // Verify stores use select chain for partition base selection (no vector-of-pointers).
    // The select and store ops interleave, so we check that at least one select on
    // ptr<3> appears and that all 4 stores are emitted.
    // Ensure no vector-of-pointers extraction is used (the old buggy pattern).
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NOT: llvm.extractelement {{.*}} !llvm.ptr<3>
    // CHECK: llvm.select {{.*}} !llvm.ptr<3>
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 512 : i32, 66048 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}
