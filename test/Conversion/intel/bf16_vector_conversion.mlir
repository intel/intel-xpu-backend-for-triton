// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// f32 -> bf16 RTNE conversions use the 8-wide __spirv_ConvertFToBF16INTEL
// wherever ttig.support_bfloat16_conversion allows the scalar builtin, in full
// groups of 8 only.

#b4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#b8 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#b16 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"ttig.support_bfloat16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: llvm.func spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f(vector<8xf32>) -> vector<8xi16>

  // One group: lane i of the call holds element i, and result i comes from lane i.
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_vector(
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         %[[A0:.*]] = llvm.extractvalue %arg0[0]
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         %[[A7:.*]] = llvm.extractvalue %arg0[7]
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT:    %[[V0:.*]] = llvm.insertelement %[[A0]], %{{.*}}[%[[C0]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    %[[V1:.*]] = llvm.insertelement %{{.*}}, %[[V0]][%[[C1]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    %[[V2:.*]] = llvm.insertelement %{{.*}}, %[[V1]][%[[C2]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT:    %[[V3:.*]] = llvm.insertelement %{{.*}}, %[[V2]][%[[C3]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:    %[[V4:.*]] = llvm.insertelement %{{.*}}, %[[V3]][%[[C4]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK-NEXT:    %[[V5:.*]] = llvm.insertelement %{{.*}}, %[[V4]][%[[C5]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C6:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK-NEXT:    %[[V6:.*]] = llvm.insertelement %{{.*}}, %[[V5]][%[[C6]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK-NEXT:    %[[V7:.*]] = llvm.insertelement %[[A7]], %[[V6]][%[[C7]] : i32] : vector<8xf32>
  // CHECK-NEXT:    %[[CVT:.*]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f(%[[V7]])
  // CHECK-NEXT:    %[[D0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT:    %[[E0:.*]] = llvm.extractelement %[[CVT]][%[[D0]] : i32] : vector<8xi16>
  // CHECK-NEXT:    %[[B0:.*]] = llvm.bitcast %[[E0]] : i16 to bf16
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         %[[D7:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK-NEXT:    %[[E7:.*]] = llvm.extractelement %[[CVT]][%[[D7]] : i32] : vector<8xi16>
  // CHECK-NEXT:    %[[B7:.*]] = llvm.bitcast %[[E7]] : i16 to bf16
  // CHECK-NEXT:    llvm.mlir.undef
  // CHECK-NEXT:    llvm.insertvalue %[[B0]], %{{.*}}[0]
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.insertvalue %[[B7]], %{{.*}}[7]
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.return
  tt.func @truncf_vector(%arg0: tensor<128xf32, #b8>) -> tensor<128xbf16, #b8> {
    %0 = arith.truncf %arg0 : tensor<128xf32, #b8> to tensor<128xbf16, #b8>
    tt.return %0 : tensor<128xbf16, #b8>
  }

  // Fewer than 8 elements per thread: scalar calls only.
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_below_width(
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         @_Z27__spirv_ConvertFToBF16INTELf({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         @_Z27__spirv_ConvertFToBF16INTELf({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         @_Z27__spirv_ConvertFToBF16INTELf({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         @_Z27__spirv_ConvertFToBF16INTELf({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         llvm.return
  tt.func @truncf_below_width(%arg0: tensor<64xf32, #b4>) -> tensor<64xbf16, #b4> {
    %0 = arith.truncf %arg0 : tensor<64xf32, #b4> to tensor<64xbf16, #b4>
    tt.return %0 : tensor<64xbf16, #b4>
  }

  // Two groups: exactly two vector calls and no scalar ones.
  // CHECK-LABEL: llvm.func spir_kernelcc @fp_to_fp_rtne(
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELDv8_f({{.*}}
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.return
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
  // CHECK-COUNT-8: @_Z27__spirv_ConvertFToBF16INTELf(
  // CHECK-NOT:     ConvertFToBF16INTELDv8_f
  // CHECK:         llvm.return
  tt.func @truncf_f64(%arg0: tensor<128xf64, #b8>) -> tensor<128xbf16, #b8> {
    %0 = arith.truncf %arg0 : tensor<128xf64, #b8> to tensor<128xbf16, #b8>
    tt.return %0 : tensor<128xbf16, #b8>
  }
}

// -----

// No conversion support: the software sequence, no builtin.
#b16 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @truncf_no_support(
  // CHECK-NOT:     ConvertFToBF16INTEL
  // CHECK:         llvm.return
  tt.func @truncf_no_support(%arg0: tensor<256xf32, #b16>) -> tensor<256xbf16, #b16> {
    %0 = arith.truncf %arg0 : tensor<256xf32, #b16> to tensor<256xbf16, #b16>
    tt.return %0 : tensor<256xbf16, #b16>
  }
}
