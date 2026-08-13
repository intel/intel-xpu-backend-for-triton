// RUN: triton-opt %s -split-input-file -convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 16], order = [1, 0]}>
#linear = #ttg.linear<{register = [[32, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[64, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = #ttg.cga<{CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @local_load_to_linear(%arg0: tensor<128x256xbf16, #blocked>) {
    // CHECK-LABEL: llvm.func @local_load_to_linear
    // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}xbf16>
    %0 = ttg.local_alloc %arg0 : (tensor<128x256xbf16, #blocked>) -> !ttg.memdesc<128x256xbf16, #shared, #smem>
    %1 = ttg.local_load %0 : !ttg.memdesc<128x256xbf16, #shared, #smem> -> tensor<128x256xbf16, #linear>
    tt.return
  }
}
