// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// The cross-warp reduction reads its result back from the shared-memory
// scratch buffer, which the next shared-memory user overwrites after the next
// work-group barrier. The read must be volatile: with plain loads the code
// IGC generates for two such reductions added together is wrong at 2 warps x
// 16 lanes (intel-xpu-backend-for-triton #8487).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @two_row_sums
  tt.func public @two_row_sums(%a: tensor<16x128xf32, #blocked>, %b: tensor<16x128xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = "tt.reduce"(%b) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = arith.addf %0, %1 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // The 16 results of each of the two reductions are read back volatile.
    // CHECK-COUNT-32: llvm.load volatile %{{.*}} : !llvm.ptr<3> -> f32
    // CHECK-NOT: llvm.load volatile
    tt.return %2 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}
