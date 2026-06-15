// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that precise sqrt and divf lower to OpenCL sqrt builtin/llvm.fdiv
// COM: with FPRoundingMode attribute when ttig.support_rounded_divide_sqrt is present.
module attributes {"ttg.num-warps" = 4 : i32, ttig.support_rounded_divide_sqrt} {
  // CHECK-LABEL: @precise_math_rounded_divide_sqrt
  tt.func public @precise_math_rounded_divide_sqrt(%arg0: tensor<128xf32, #blocked>, %arg1: tensor<128xf32, #blocked>) {
    // CHECK: llvm.call spir_funccc @_Z16__spirv_ocl_sqrtf({{.*}}) {{{.*}}triton_gen.FPRoundingMode = 0 : i32{{.*}}}
    %0 = tt.precise_sqrt %arg0 : tensor<128xf32, #blocked>
    // CHECK: llvm.fdiv {{.*}} {triton_gen.FPRoundingMode = 0 : i32}
    %1 = tt.precise_divf %arg0, %arg1 : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that precise sqrt and divf lower to __imf_* f32 builtins
// COM: when ttig.support_rounded_divide_sqrt is absent.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @precise_math_imf_fallback_f32
  tt.func public @precise_math_imf_fallback_f32(%arg0: tensor<128xf32, #blocked>, %arg1: tensor<128xf32, #blocked>) {
    // CHECK: llvm.call spir_funccc @__imf_sqrtf_rn
    %0 = tt.precise_sqrt %arg0 : tensor<128xf32, #blocked>
    // CHECK: llvm.call spir_funccc @__imf_fdiv_rn
    %1 = tt.precise_divf %arg0, %arg1 : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that precise sqrt and divf lower to __imf_* f64 builtins
// COM: when ttig.support_rounded_divide_sqrt is absent.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @precise_math_imf_fallback_f64
  tt.func public @precise_math_imf_fallback_f64(%arg0: tensor<128xf64, #blocked>, %arg1: tensor<128xf64, #blocked>) {
    // CHECK: llvm.call spir_funccc @__imf_sqrt_rn
    %0 = tt.precise_sqrt %arg0 : tensor<128xf64, #blocked>
    // CHECK: llvm.call spir_funccc @__imf_ddiv_rn
    %1 = tt.precise_divf %arg0, %arg1 : tensor<128xf64, #blocked>
    tt.return
  }
}
