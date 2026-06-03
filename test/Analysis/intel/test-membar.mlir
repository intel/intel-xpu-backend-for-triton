// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// Two back-to-back sub-group transposes that share the same allocation.offset
// (both are 16-bit element types of the same shape, so the allocator reuses
// the same SLM region) but have *different* source/result element types.
// areSafeToOverlapSubGroupTransposeOps returns false at the type check, so the
// Intel membar filter must insert a barrier between them.

#blocked  = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

// CHECK-LABEL: llvm.func spir_kernelcc @test_differing_types_inserts_barrier
// CHECK: @_Z7barrierj
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_differing_types_inserts_barrier(
      %arg0: tensor<32x64xf16,  #blocked>,
      %arg1: tensor<32x64xbf16, #blocked>)
      -> (tensor<32x64xf16, #blocked1>, tensor<32x64xbf16, #blocked1>) {
    %0 = ttg.convert_layout %arg0 : tensor<32x64xf16,  #blocked> -> tensor<32x64xf16,  #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<32x64xbf16, #blocked> -> tensor<32x64xbf16, #blocked1>
    tt.return %0, %1 : tensor<32x64xf16, #blocked1>, tensor<32x64xbf16, #blocked1>
  }
}
