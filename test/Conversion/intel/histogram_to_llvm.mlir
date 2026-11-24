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

// -----

// COM: The result layout has `sizePerThread = [4]` on a shape-[2] tensor, so its
// COM: register dimension carries zero bases: getUniqueElemsPerThread is 2 while
// COM: getTotalElemsPerThread is 4. The lowering must emit one index per *unique*
// COM: register, otherwise it builds 4 values for a 2-element LLVM struct and
// COM: aborts with "size mismatch when packing elements for LLVM struct".
// COM: The 2-element result struct below is what pins this.

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @histogram_register_broadcast
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i32, i32)>
  tt.func @histogram_register_broadcast(%src: tensor<256xi32, #src>, %mask: tensor<256xi1, #src>) -> tensor<2xi32, #dst> {
    %hist = tt.histogram %src, %mask : tensor<256xi32, #src> -> tensor<2xi32, #dst>
    tt.return %hist : tensor<2xi32, #dst>
  }
}
