// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

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
