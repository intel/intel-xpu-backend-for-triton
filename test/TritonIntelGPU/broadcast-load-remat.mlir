// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: Verify that a broadcast load (bias vector broadcast to a 2D tile) is
// COM: not treated as an expensive anchor.  The RemoveLayoutConversions pass
// COM: should rematerialize the splat->addptr->broadcast->load chain in the
// COM: target (#mma) layout, eliminating the convert_layout through SLM.

// CHECK-LABEL: @broadcast_bias_load_remat
// COM: The pass rematerializes the pointer chain in #mma layout, converting the
// COM: small 1x32 pointer tensor (cheap) then broadcasting and loading in #mma.
// COM: The critical check: the load produces #mma directly, no convert_layout
// COM: on the 32x32 data tensor after the load.
// CHECK: tt.broadcast {{.*}} -> tensor<32x32x!tt.ptr<f16>, #mma>
// CHECK-NEXT: tt.load {{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
// CHECK-NEXT: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // The bias load pattern: splat -> addptr -> broadcast -> load (32x32 but
  // only 32 unique elements along the column dimension).
  // With our fix, effective elements = 32 < numWarps * threadsPerWarp = 64,
  // so the load is inexpensive and the convert_layout can be eliminated.
  tt.func public @broadcast_bias_load_remat(%bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %col_offsets: tensor<1x32xi32, #blocked>) -> tensor<32x32xf16, #mma> {
    %splat = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked>
    %addptr = tt.addptr %splat, %col_offsets : tensor<1x32x!tt.ptr<f16>, #blocked>, tensor<1x32xi32, #blocked>
    %bcast = tt.broadcast %addptr : tensor<1x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %load = tt.load %bcast : tensor<32x32x!tt.ptr<f16>, #blocked>
    %cvt = ttg.convert_layout %load : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #mma>
    tt.return %cvt : tensor<32x32xf16, #mma>
  }
}

// -----

// COM: Negative test: a non-broadcast load with 1024 elements (>= 64 threads)
// COM: should remain expensive and the convert_layout should be preserved.

// CHECK-LABEL: @non_broadcast_load_stays_expensive
// CHECK: tt.load {{.*}} : tensor<32x32x!tt.ptr<f16>, #blocked>
// CHECK-NEXT: ttg.convert_layout

#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // A regular 2D load with no broadcast — all 1024 elements are unique.
  // This should remain expensive (no rematerialization).
  tt.func public @non_broadcast_load_stays_expensive(%ptr: tensor<32x32x!tt.ptr<f16>, #blocked2>) -> tensor<32x32xf16, #mma2> {
    %load = tt.load %ptr : tensor<32x32x!tt.ptr<f16>, #blocked2>
    %cvt = ttg.convert_layout %load : tensor<32x32xf16, #blocked2> -> tensor<32x32xf16, #mma2>
    tt.return %cvt : tensor<32x32xf16, #mma2>
  }
}
