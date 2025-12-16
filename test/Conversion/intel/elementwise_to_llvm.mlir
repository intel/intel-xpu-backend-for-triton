// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_bfloat16_arithmetic, ttig.target_arch = "spir64"} {
  tt.func public @kernel(%X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    // CHECK: llvm.intr.fabs({{.*}}) : (bf16) -> bf16
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %x = tt.splat %X : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked>
    %x_0 = tt.addptr %x, %off : tensor<128x!tt.ptr<bf16>, #blocked>, tensor<128xi32, #blocked>
    %x_1 = tt.load %x_0 : tensor<128x!tt.ptr<bf16>, #blocked>
    %z = math.absf %x_1 : tensor<128xbf16, #blocked>
    tt.return
  }
}
