// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Software fp8e4m3(OCP e4m3fn) -> fp16 upcast (oneDNN 6-op sequence: ashr, and, bitcast,
// COM: fmul x3, fadd). Module lacks ttig.support_f8_conversion, so the gate selects the
// COM: software path instead of the SPIR-V builtin (see fp8_convert.mlir for the gated path).
// COM: This pins the exact op sequence, in particular the trailing fmul+fadd pair, so a future
// COM: InstCombine/fast-math regression cannot silently delete the Inf->NaN fixup -- which
// COM: would turn the reserved bytes 0x7F/0xFF into a plausible-looking finite number instead
// COM: of NaN.
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16
  tt.func public @convert_fp8e4m3_to_fp16(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(3.686400e+04 : f16) : f16
    // CHECK-DAG: llvm.mlir.constant(6.942750e-03 : f16) : f16
    // CHECK-DAG: llvm.mlir.constant(0.000000e+00 : f16) : f16
    // CHECK-DAG: llvm.mlir.constant(-16385 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(1 : i16) : i16
    // CHECK: llvm.ashr {{.*}} : vector<2xi16>
    // CHECK: llvm.and {{.*}} : vector<2xi16>
    // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to vector<2xf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xf16>
    // CHECK: llvm.fadd {{.*}} : vector<2xf16>
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<16xf16, #blocked>
  }
}
