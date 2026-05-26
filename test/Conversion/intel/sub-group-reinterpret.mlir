// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// 32x32xf8E5M2 mma -> dot_a layout conversion via sub-group bitcast shuffle.
// The conversion reinterprets packed f8E5M2 elements by calling the
// GenISA_SubgroupBitcastShuffle intrinsic.

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v2i16.v4i8(vector<4xi8>) -> vector<2xi16>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_reinterpret(
  tt.func @test_reinterpret(%arg0: tensor<32x32xf8E5M2, #mma>) -> tensor<32x32xf8E5M2, #dot_a> {
    // COM: f8E5M2 elements are bitcast to i8 for the shuffle operation.
    // CHECK-COUNT-64: llvm.bitcast {{.*}} : f8E5M2 to i8
    // COM: The GenISA bitcast shuffle intrinsic is used to reinterpret 4xi8 -> 2xi16.
    // CHECK-COUNT-16: llvm.call spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v2i16.v4i8(
    // COM: Elements are bitcast back to f8E5M2 after the shuffle.
    // CHECK-COUNT-64: llvm.bitcast {{.*}} : i8 to f8E5M2
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf8E5M2, #mma> -> tensor<32x32xf8E5M2, #dot_a>
    tt.return %0 : tensor<32x32xf8E5M2, #dot_a>
  }
}
