// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that f32 math.sin lowers to llvm.sin.f32 with the fast fastmath flag
// COM: when ttig.fast_math is set on the module.
module attributes {"ttg.num-warps" = 4 : i32, "ttig.fast_math"} {
  // CHECK-LABEL: @sin_f32_fast_math
  // CHECK-NOT: @_Z15__spirv_ocl_sinf
  tt.func public @sin_f32_fast_math(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: llvm.call @llvm.sin.f32
    // CHECK-SAME: fastmathFlags = #llvm.fastmath<fast>
    %0 = math.sin %arg0 : tensor<128xf32, #blocked>
    tt.return %0 : tensor<128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that f32 math.cos lowers to llvm.cos.f32 with the fast fastmath flag
// COM: when ttig.fast_math is set on the module.
module attributes {"ttg.num-warps" = 4 : i32, "ttig.fast_math"} {
  // CHECK-LABEL: @cos_f32_fast_math
  // CHECK-NOT: @_Z15__spirv_ocl_cosf
  tt.func public @cos_f32_fast_math(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: llvm.call @llvm.cos.f32
    // CHECK-SAME: fastmathFlags = #llvm.fastmath<fast>
    %0 = math.cos %arg0 : tensor<128xf32, #blocked>
    tt.return %0 : tensor<128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that f32 math.sin falls through to the precise SPIR-V builtin when
// COM: ttig.fast_math is NOT set (default path — no precision degradation).
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @sin_f32_no_fast_math
  // CHECK-NOT: @llvm.sin.f32
  tt.func public @sin_f32_no_fast_math(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: llvm.call spir_funccc @_Z15__spirv_ocl_sinf
    %0 = math.sin %arg0 : tensor<128xf32, #blocked>
    tt.return %0 : tensor<128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// COM: Test that f16 math.sin falls through to the vendor library regardless of
// COM: fast_math — the SinOpConversionApprox pattern is f32-only.
module attributes {"ttg.num-warps" = 4 : i32, "ttig.fast_math"} {
  // CHECK-LABEL: @sin_f16_vendor_lib
  // CHECK-NOT: @llvm.sin.f32
  tt.func public @sin_f16_vendor_lib(%arg0: tensor<128xf16, #blocked>) -> tensor<128xf16, #blocked> {
    // CHECK: llvm.call spir_funccc @_Z15__spirv_ocl_sinf
    %0 = math.sin %arg0 : tensor<128xf16, #blocked>
    tt.return %0 : tensor<128xf16, #blocked>
  }
}
