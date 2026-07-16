// RUN: env TRITON_INTEL_PREDICATED_LOAD=0 TRITON_INTEL_PREDICATED_STORE=0 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// COM: Regression test for a fused reduction + scatter (atomic_rmw) miscompile
//      (multi_margin_loss backward, issue #7436). When a layout places several
//      registers of the same value in one thread (register redundancy, e.g. a
//      broadcasted reduction result), the atomic must be emitted only ONCE per
//      thread. Emitting it once per redundant register double-counts the
//      atomic_add and produces a 2x (or Nx) result.

// COM: sizePerThread = [1, 4] over a 16x1 tensor gives 4 registers per thread
//      that all map to the same tensor element -> 3 of the 4 are redundant.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_rmw_reg_redundant
  // CHECK-COUNT-1: llvm.atomicrmw fadd
  // CHECK-NOT: llvm.atomicrmw fadd
  tt.func public @atomic_rmw_reg_redundant(%ptr: !tt.ptr<f32>, %val: f32) {
    %mask = arith.constant dense<true> : tensor<16x1xi1, #blocked>
    %ptr_splat = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %range_exp = tt.expand_dims %range {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range_exp : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %v = tt.splat %val : f32 -> tensor<16x1xf32, #blocked>
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %v, %mask : (tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xf32, #blocked>, tensor<16x1xi1, #blocked>) -> tensor<16x1xf32, #blocked>
    tt.return
  }
}
