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

// -----

// COM: Same conversion on an LTS driver (ttig.is_lts), which selects the 11-op
// COM: integer-domain sequence instead: the oneDNN sequence is short, unpredicated
// COM: float arithmetic that the LTS IGC's LoopSink clones ~6x inside unrolled
// COM: loops, tripling compile time on fp8 GEMMs (issue #8046). The icmp/select
// COM: NaN fixup here is what keeps this version from being sunk, so this test
// COM: pins it along with the single exact x256 rebias.
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.is_lts, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16_lts
  tt.func public @convert_fp8e4m3_to_fp16_lts(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(2.560000e+02 : f16) : f16
    // CHECK-DAG: llvm.mlir.constant(8323199 : i32) : i32
    // CHECK: llvm.lshr {{.*}} : i32
    // CHECK: llvm.and {{.*}} : i32
    // CHECK: llvm.shl {{.*}} : i32
    // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xf16>
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.select
    // CHECK: llvm.select
    // COM: The oneDNN sequence must not be used here.
    // CHECK-NOT: llvm.ashr {{.*}} : vector<2xi16>
    // CHECK-NOT: llvm.fadd {{.*}} : vector<2xf16>
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<16xf16, #blocked>
  }
}
