// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Software fp8e4m3(OCP e4m3fn) -> fp16 upcast: sext, shl, and, bitcast, one exact
// COM: fmul x256 and a fabs/fcmp/select that turns the reserved bytes 0x7F/0xFF (which
// COM: the multiply maps to +-480, the only magnitude above the largest finite value 448)
// COM: into NaN. Module lacks ttig.support_f8_conversion, so the gate selects the
// COM: software path instead of the SPIR-V builtin (see fp8_convert.mlir for the gated path).
// COM: This pins the exact op sequence, in particular the trailing fcmp/select pair, so a
// COM: regression in this lowering or its canonicalization cannot silently delete the NaN
// COM: fixup -- which would turn the reserved bytes 0x7F/0xFF into a plausible-looking
// COM: finite number instead of NaN.
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16
  tt.func public @convert_fp8e4m3_to_fp16(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xf16, #blocked>
    // CHECK-DAG: [[C256:%.*]] = llvm.mlir.constant(2.560000e+02 : f16) : f16
    // CHECK-DAG: [[C448:%.*]] = llvm.mlir.constant(4.480000e+02 : f16) : f16
    // CHECK-DAG: [[QNAN:%.*]] = llvm.mlir.constant(32256 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(-16385 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(7 : i16) : i16
    // CHECK-NOT: vector<4xi8>
    // CHECK: llvm.sext {{.*}} : i8 to i16
    // CHECK: llvm.shl {{.*}} : i16
    // CHECK: llvm.and {{.*}} : i16
    // CHECK: llvm.bitcast {{.*}} : i16 to f16
    // CHECK: [[M:%.*]] = llvm.fmul {{.*}}, [[C256]] : f16
    // CHECK-NEXT: [[A:%.*]] = llvm.intr.fabs([[M]]) : (f16) -> f16
    // CHECK-NEXT: [[C:%.*]] = llvm.fcmp "ogt" [[A]], [[C448]] : f16
    // CHECK-NEXT: [[N:%.*]] = llvm.bitcast [[QNAN]] : i16 to f16
    // CHECK-NEXT: llvm.select [[C]], [[N]], [[M]] : i1, f16
    // CHECK-NOT: vector<4xi8>
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<16xf16, #blocked>
  }
}

// -----

// COM: Same conversion on an LTS driver (ttig.is_lts), which selects the 11-op
// COM: integer-domain sequence instead: the previous oneDNN-derived float sequence was
// COM: short, unpredicated arithmetic that the LTS IGC's LoopSink cloned ~6x inside
// COM: unrolled loops, tripling compile time on fp8 GEMMs (issue #8046). The icmp/select
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
    // COM: The float-domain sequence must not be used here.
    // CHECK-NOT: llvm.sext {{.*}} : i8 to i16
    // CHECK-NOT: llvm.fcmp
    // CHECK-NOT: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE4M3ToFP16INTEL
    tt.return %dst : tensor<16xf16, #blocked>
  }
}
