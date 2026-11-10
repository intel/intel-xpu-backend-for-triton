// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// f32 -> bf16 RTNE conversions use the 8-wide __spirv_ConvertFToBF16INTEL when
// ttig.support_vector_bf16_conversion is set, in full groups of 8 only.

#b16 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#b4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"ttig.support_bfloat16_conversion", "ttig.support_vector_bf16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: llvm.func spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f(vector<8xf32>) -> vector<8xi16>

  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_vector(
  // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<8xf32>
  // CHECK:         %[[CVT:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f(%{{.*}}) {{.*}} : (vector<8xf32>) -> vector<8xi16>
  // CHECK:         %[[E0:.*]] = llvm.extractelement %[[CVT]][%{{.*}} : i32] : vector<8xi16>
  // CHECK:         llvm.bitcast %[[E0]] : i16 to bf16
  // CHECK:         llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f
  // CHECK-NOT:     @_Z27__spirv_ConvertFToBF16INTELf(
  // CHECK:         llvm.return
  tt.func @truncf_vector(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = arith.truncf %arg0 : tensor<256xf32, #b16> to tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }

  // Fewer than 8 elements per thread: scalar calls only.
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_below_width(
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK-COUNT-4: @_Z27__spirv_ConvertFToBF16INTELf(
  tt.func @truncf_below_width(%arg0: tensor<64xf32, #b4>) -> tensor<64xbf16, #b4> {
    %0 = arith.truncf %arg0 : tensor<64xf32, #b4> to tensor<64xbf16, #b4>
    tt.return %0 : tensor<64xbf16, #b4>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @fp_to_fp_rtne(
  // CHECK-COUNT-2: @_Z27__spirv_ConvertFToBF16INTELDv8_f(
  tt.func @fp_to_fp_rtne(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<256xf32, #b16> -> tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }

  // Non-RTNE rounding keeps the software sequence.
  // CHECK-LABEL: llvm.func spir_kernelcc @fp_to_fp_rtz(
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.return
  tt.func @fp_to_fp_rtz(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<256xf32, #b16> -> tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }

  // f64 keeps its round-to-odd step and the scalar call.
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_f64(
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         @_Z27__spirv_ConvertFToBF16INTELf(
  tt.func @truncf_f64(%arg0: tensor<256xf64, #b16>) -> tensor<256xbf16, #b16> {
    %0 = arith.truncf %arg0 : tensor<256xf64, #b16> to tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }
}

// -----

// Gate off: one scalar call per element.
#b16 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttig.support_bfloat16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_gate_off(
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK-COUNT-16: @_Z27__spirv_ConvertFToBF16INTELf(
  tt.func @truncf_gate_off(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = arith.truncf %arg0 : tensor<256xf32, #b16> to tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }
}

// -----

// Gate on without conversion support: the software sequence, no builtin.
#b16 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttig.support_vector_bf16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_no_support(
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.return
  tt.func @truncf_no_support(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = arith.truncf %arg0 : tensor<256xf32, #b16> to tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }
}
