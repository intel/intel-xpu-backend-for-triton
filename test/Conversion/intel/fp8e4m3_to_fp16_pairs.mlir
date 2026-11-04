// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --implicit-check-not=llvm.fcmp --implicit-check-not=llvm.select

// COM: A dot operand with K < 64 is upcast in pairs: both bytes of a pair in the high
// COM: halves of a <2 x i16>. The reserved bytes 0x7F/0xFF become NaN by integer ops
// COM: *before* the multiply: adding 0x80 carries through the all-ones exponent and
// COM: mantissa bits of only those bytes into bit 14, and OR-ing that bit in makes the
// COM: exponent all ones. This path has no fcmp or select, because their flag
// COM: register moves slowed down the unified attention kernels (see
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/8478).
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16_pairs
  tt.func public @convert_fp8e4m3_to_fp16_pairs(%src: tensor<32x16xf8E4M3FN, #dot1>) -> tensor<32x16xf16, #dot1> {
    // CHECK-DAG: llvm.mlir.constant(-16385 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(128 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(16384 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(2.560000e+02 : f16) : f16
    // CHECK: [[P:%.*]] = llvm.bitcast {{.*}} : vector<4xi8> to vector<2xi16>
    // CHECK: [[S:%.*]] = llvm.ashr [[P]], {{.*}} : vector<2xi16>
    // CHECK: [[A:%.*]] = llvm.and [[S]], {{.*}} : vector<2xi16>
    // CHECK: [[T:%.*]] = llvm.add [[A]], {{.*}} : vector<2xi16>
    // CHECK: [[C:%.*]] = llvm.and [[T]], {{.*}} : vector<2xi16>
    // CHECK: [[F:%.*]] = llvm.or [[A]], [[C]] : vector<2xi16>
    // CHECK: [[H:%.*]] = llvm.bitcast [[F]] : vector<2xi16> to vector<2xf16>
    // CHECK: llvm.fmul [[H]], {{.*}} : vector<2xf16>
    // CHECK-COUNT-15: llvm.fmul {{.*}} : vector<2xf16>
    %dst = tt.fp_to_fp %src : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    tt.return %dst : tensor<32x16xf16, #dot1>
  }
}
