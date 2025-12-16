// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, ttig.support_bfloat16_arithmetic} {
  tt.func public @kernel(%in: tensor<128xbf16, #blocked>) {
    // CHECK: llvm.intr.fabs({{.*}}) : (bf16) -> bf16
    %out = math.absf %in : tensor<128xbf16, #blocked>
    tt.return
  }
}
