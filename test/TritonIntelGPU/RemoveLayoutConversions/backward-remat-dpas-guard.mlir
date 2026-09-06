// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope


// COM: Backward rematerialization must not bake DPAS-family encodings into
// COM: chains of values that have no DPAS conversion path: offset (i32/i64),
// COM: mask (i1), and pointer values. The elementwise lowering only
// COM: reproduces the DPAS fan-out for this backend's dot data types (float
// COM: and 8-bit integers), so rematerializing address/mask arithmetic in a
// COM: dot_op<dpas> or (slice of) dpas layout mislowers it. The veto is per
// COM: value in the remat slice: a float chain (e.g. a descriptor load feeding
// COM: dot_op<dpas>) and the i8 chain below are still rematerialized, so the
// COM: convert_layout ops on the guarded chains must be preserved.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dpas8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot8 = #ttg.dot_op<{opIdx = 0, parent = #dpas8, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_backward_remat_dot_op_dpas_i64_chain
  tt.func @no_backward_remat_dot_op_dpas_i64_chain() -> tensor<64x32xi64, #dot0> {
    // COM: i64 offset chain feeding a convert to dot_op<dpas>: the slice
    // COM: assigns the dot_op encoding to the whole i64/i32 chain, and the
    // COM: element types are not DPAS dot data types, so the remat is skipped
    // COM: and the chain keeps its blocked encoding.
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: arith.extsi {{.*}} : tensor<64x32xi32, #blocked> to tensor<64x32xi64, #blocked>
    // CHECK: ttg.convert_layout {{.*}} : tensor<64x32xi64, #blocked> -> tensor<64x32xi64, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK: tt.return
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %b = tt.broadcast %e : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %x = arith.extsi %b : tensor<64x32xi32, #blocked> to tensor<64x32xi64, #blocked>
    %c = ttg.convert_layout %x : tensor<64x32xi64, #blocked> -> tensor<64x32xi64, #dot0>
    tt.return %c : tensor<64x32xi64, #dot0>
  }

  // CHECK-LABEL: @no_backward_remat_dpas_slice_i64_chain
  tt.func @no_backward_remat_dpas_slice_i64_chain() -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dpas}>> {
    // COM: i64 chain targeting slice<dpas>: the slice unwraps to a DPAS
    // COM: encoding and i64 is not a DPAS dot data type, so the remat is
    // COM: skipped and the convert stays.
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: arith.extsi {{.*}} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{.*}} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #mma}>>
    // CHECK: tt.return
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %x = arith.extsi %r : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c = ttg.convert_layout %x : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dpas}>>
    tt.return %c : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dpas}>>
  }

  // CHECK-LABEL: @no_backward_remat_dpas_slice_i1_mask_chain
  tt.func @no_backward_remat_dpas_slice_i1_mask_chain(%arg0: i32) -> tensor<64xi1, #ttg.slice<{dim = 1, parent = #dpas}>> {
    // COM: i1 mask chain (cmpi against a splat argument so it cannot be
    // COM: constant-folded) targeting slice<dpas>: i1 is not a DPAS dot data
    // COM: type, convert stays.
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: arith.cmpi {{.*}} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{.*}} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi1, #ttg.slice<{dim = 1, parent = #mma}>>
    // CHECK: tt.return
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %s = tt.splat %arg0 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m = arith.cmpi slt, %r, %s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c = ttg.convert_layout %m : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi1, #ttg.slice<{dim = 1, parent = #dpas}>>
    tt.return %c : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dpas}>>
  }

  // CHECK-LABEL: @backward_remat_dpas_slice_f32_chain_allowed
  tt.func public @backward_remat_dpas_slice_f32_chain_allowed(%arg0: f32) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dpas}>> {
    // COM: Positive control: an all-dot-data-type chain (splat/mulf of f32)
    // COM: has no value outside the DPAS dot data types in the remat slice, so
    // COM: rematerializing it in a slice<DPAS> layout is sound and the convert
    // COM: is still removed. (The i32 make_range feeding a sitofp is NOT a dot
    // COM: data type, so that variant is governed by the vetoed cases above.)
    // CHECK: tt.splat {{.*}} : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    // CHECK: arith.mulf {{.*}} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return
    %f = tt.splat %arg0 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %x = arith.mulf %f, %f : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c = ttg.convert_layout %x : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dpas}>>
    tt.return %c : tensor<64xf32, #ttg.slice<{dim = 1, parent = #dpas}>>
  }

  // CHECK-LABEL: @backward_remat_dot_op_dpas_i8_chain_allowed
  tt.func public @backward_remat_dot_op_dpas_i8_chain_allowed(%arg0: i8) -> tensor<64x32xi8, #dot8> {
    // COM: Positive control: an i8-only chain (splat/addi) feeding a convert
    // COM: to dot_op<dpas>. i8 is a first-class DPAS dot data type (packed
    // COM: four per channel, opsPerChan = 4), so the remat is not vetoed and
    // COM: the convert is removed.
    // CHECK: tt.splat {{.*}} : i8 -> tensor<64x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
    // CHECK: arith.addi {{.*}} : tensor<64x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return
    %s = tt.splat %arg0 : i8 -> tensor<64x32xi8, #blocked>
    %x = arith.addi %s, %s : tensor<64x32xi8, #blocked>
    %c = ttg.convert_layout %x : tensor<64x32xi8, #blocked> -> tensor<64x32xi8, #dot8>
    tt.return %c : tensor<64x32xi8, #dot8>
  }
}
