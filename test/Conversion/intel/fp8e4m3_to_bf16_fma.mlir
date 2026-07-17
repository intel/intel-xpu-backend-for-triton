// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Non-LTS driver path: fast fmul BF16 converter. Module has ttig.support_bfloat16_arithmetic,
// COM: so the gate selects the i16-mask + fmul bf16 path (2^120 multiplier, no integer table).
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64", ttig.support_bfloat16_arithmetic } {
  // CHECK-LABEL: @convert_fp8e4m3_to_bf16_fmul
  tt.func public @convert_fp8e4m3_to_bf16_fmul(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(2032 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(4 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(1.329230e+36 : bf16) : bf16
    // CHECK: llvm.trunc {{.*}} : i32 to i16
    // CHECK: llvm.lshr {{.*}} : i16
    // CHECK: llvm.and {{.*}} : i16
    // CHECK: llvm.fmul {{.*}} : bf16
    // CHECK-NOT: llvm.mlir.constant(260046848 : i32)
    // CHECK-NOT: llvm.mlir.constant(1006632960 : i32)
    // CHECK-NOT: llvm.mlir.constant(989855744 : i32)
    // CHECK-NOT: llvm.extractelement {{.*}}vector<8xi32>
    // CHECK-NOT: llvm.icmp "eq"
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}

// -----

// COM: LTS driver path: pure-integer table-lookup converter. Module lacks ttig.support_bfloat16_arithmetic,
// COM: so the gate selects the fallback (subnormal detection + 8-element table + rebias add, no bf16 fmul).
#blocked1 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  // CHECK-LABEL: @convert_fp8e4m3_to_bf16_table
  tt.func public @convert_fp8e4m3_to_bf16_table(%src: tensor<16xf8E4M3FN, #blocked1>) -> tensor<16xbf16, #blocked1> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked1> -> tensor<16xbf16, #blocked1>
    // CHECK-DAG: llvm.mlir.constant(260046848 : i32)
    // CHECK-DAG: llvm.mlir.constant(1006632960 : i32)
    // CHECK-DAG: llvm.mlir.constant(989855744 : i32)
    // CHECK: llvm.icmp "eq" {{.*}} : i32
    // CHECK: llvm.extractelement {{.*}}vector<8xi32>
    // CHECK: llvm.add {{.*}} : i32
    // CHECK-NOT: llvm.mlir.constant(2032 : i16)
    // CHECK-NOT: llvm.mlir.constant(4 : i16)
    // CHECK-NOT: llvm.mlir.constant(1.329230e+36 : bf16)
    // CHECK-NOT: llvm.fmul
    tt.return %dst : tensor<16xbf16, #blocked1>
  }
}
