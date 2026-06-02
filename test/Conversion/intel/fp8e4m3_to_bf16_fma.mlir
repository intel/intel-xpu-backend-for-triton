// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert_fp8e4m3_to_bf16(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(2032 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(4 : i16) : i16
    // CHECK: llvm.trunc {{.*}} : i32 to i16
    // CHECK: llvm.lshr {{.*}} : i16
    // CHECK: llvm.and {{.*}} : i16
    // CHECK: llvm.fmul {{.*}} : bf16
    // CHECK-NOT: llvm.and {{.*}}133169152
    // CHECK-NOT: llvm.add {{.*}}1006632960
    // CHECK-NOT: llvm.extractelement {{.*}}<i32 0, i32 989855744
    // CHECK-NOT: llvm.icmp{{.*}}eq{{.*}}0xf800000
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}
