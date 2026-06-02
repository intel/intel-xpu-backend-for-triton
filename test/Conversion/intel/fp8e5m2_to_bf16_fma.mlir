// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Non-LTS path — fast fmul-based converter (requires ttig.support_bfloat16_arithmetic)
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_arithmetic, ttig.target_arch = "spir64" } {
  // CHECK-LABEL: @convert_fp8e5m2_to_bf16_fmul
  tt.func public @convert_fp8e5m2_to_bf16_fmul(%src: tensor<16xf8E5M2, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E5M2, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(5.192300e+33 : bf16) : bf16
    // CHECK-DAG: llvm.mlir.constant(2147450879 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(-2147450880 : i32) : i32
    // CHECK: llvm.and {{.*}} : i32
    // CHECK: llvm.lshr {{.*}} : i32
    // CHECK: llvm.and {{.*}} : i32
    // CHECK: llvm.or {{.*}} : i32
    // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xbf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xbf16>
    // CHECK-NOT: llvm.extractelement {{.*}}vector<4xi32>
    // CHECK-NOT: llvm.icmp "eq"
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}

// -----

// COM: LTS path — pure-integer table-lookup fallback (no ttig.support_bfloat16_arithmetic)
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  // CHECK-LABEL: @convert_fp8e5m2_to_bf16_table
  tt.func public @convert_fp8e5m2_to_bf16_table(%src: tensor<16xf8E5M2, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E5M2, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(260046848 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(939524096 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(931135488 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(943718400 : i32) : i32
    // CHECK: llvm.icmp "eq" {{.*}} : i32
    // CHECK: llvm.extractelement {{.*}}vector<4xi32>
    // CHECK: llvm.add {{.*}} : i32
    // CHECK: llvm.select {{.*}} : i1, i32
    // CHECK-NOT: llvm.fmul
    // CHECK-NOT: llvm.mlir.constant(5.192300e+33 : bf16)
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}
