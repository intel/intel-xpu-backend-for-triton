// RUN: triton-opt %s -split-input-file --triton-annotate-module='support-f8-conversion' --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --check-prefixes=CHECK-FP8-CONV
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<16xf16, #blocked>) -> tensor<16xf8E4M3FN, #blocked> {
    %dst = tt.fp_to_fp %src, rounding = rtne : tensor<16xf16, #blocked> -> tensor<16xf8E4M3FN, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDv16_Dh
    // CHECK-NOT: llvm.call spir_funccc @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTEL
    tt.return %dst : tensor<16xf8E4M3FN, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<8xf16, #blocked>) -> tensor<8xf8E5M2, #blocked> {
    %dst = tt.fp_to_fp %src, rounding = rtne : tensor<8xf16, #blocked> -> tensor<8xf8E5M2, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDv8_Dh
    // CHECK-NOT: llvm.call spir_funccc @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTEL
    tt.return %dst : tensor<8xf8E5M2, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<4xf8E5M2, #blocked>) -> tensor<4xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<4xf8E5M2, #blocked> -> tensor<4xf16, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE5M2ToFP16INTELDv4_c
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE5M2ToFP16INTEL
    tt.return %dst : tensor<4xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<2xf8E4M3FN, #blocked>) -> tensor<2xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<2xf8E4M3FN, #blocked> -> tensor<2xf16, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTELDv2_c
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<2xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<1xf8E4M3FN, #blocked>) -> tensor<1xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<1xf8E4M3FN, #blocked> -> tensor<1xf16, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTELc
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<1xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<32xf8E4M3FN, #blocked>) -> tensor<32xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<32xf8E4M3FN, #blocked> -> tensor<32xf16, #blocked>
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTELDv16_c
    // CHECK-FP8-CONV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTELDv16_c
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<32xf16, #blocked>
  }
}
