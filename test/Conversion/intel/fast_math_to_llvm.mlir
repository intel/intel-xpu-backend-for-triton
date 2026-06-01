// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that fast_sinf lowers to the __imf_sinf_la IMF low-accuracy builtin.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @fast_sinf_to_llvm
  tt.func public @fast_sinf_to_llvm(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: llvm.call spir_funccc @__imf_sinf_la
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__imf_sinf_la"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return %0 : tensor<128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that fast_cosf lowers to the __imf_cosf_la IMF low-accuracy builtin.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @fast_cosf_to_llvm
  tt.func public @fast_cosf_to_llvm(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: llvm.call spir_funccc @__imf_cosf_la
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__imf_cosf_la"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return %0 : tensor<128xf32, #blocked>
  }
}
