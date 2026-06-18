// RUN: env TRITON_INTEL_PREDICATED_LOAD=0 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,NO-PREDICATED
// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,PREDICATED
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,DEFAULT

// COM: Test scalar PredicatedLoadINTEL behavior with different env var settings.
// COM: By default, scalar (non-vectorized) masked loads use the control-flow
// COM: fallback instead of PredicatedLoadINTEL due to #7235.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: scalar_masked_load
  tt.func @scalar_masked_load(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    // With vec=1 (mask from function arg has alignment 1), loads are scalar.
    // PREDICATED: triton_gen.predicated_load {{.*}} : (!llvm.ptr<1>, i1, i32) -> i32
    // NO-PREDICATED-NOT: triton_gen.predicated_load
    // DEFAULT-NOT: triton_gen.predicated_load
    %val = tt.load %ptr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
