// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Verify that the Intel histogram lowering produces atomic-per-element
// COM: code (atomicrmw add) and does NOT produce ballot/ctpop instructions.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @histogram_atomic
  // CHECK: llvm.icmp "ult"
  // CHECK: llvm.atomicrmw add
  // CHECK-NOT: ballot
  // CHECK-NOT: ctpop
  tt.func @histogram_atomic(%src: tensor<256xi32, #blocked>, %mask: tensor<256xi1, #blocked>, %out_ptr: tensor<8x!tt.ptr<i32>, #blocked>) {
    %hist = tt.histogram %src, %mask : tensor<256xi32, #blocked> -> tensor<8xi32, #blocked>
    tt.store %out_ptr, %hist : tensor<8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
