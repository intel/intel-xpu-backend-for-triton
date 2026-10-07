// RUN: triton-opt %s --tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: visitDescriptor stamps ttig.block_io = "row_major" covering EVERY
// COM: candidate, so every candidate must have a unit last stride. The IR is
// COM: valid either way: tt.make_tensor_descriptor carries no unit-stride
// COM: invariant (only Pure + SameVariadicOperandSize) and the descriptor
// COM: load/store verifier checks element type and total element count, nothing
// COM: about stride values. A non-unit last stride must therefore leave the op
// COM: to the gather lowering -- no block_io, but still ttig.desc_padding, which
// COM: the load lowering reads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // COM: Candidate order matters here. The worklist pushes the then yield and
  // COM: then the else yield, and pops from the back, so the ELSE branch's
  // COM: descriptor is candidate 0. Giving the else branch stride 1 and the then
  // COM: branch stride 2 means a check of candidate 0 alone would pass.
  // CHECK-LABEL: tt.func @if_divergent_last_stride
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.descriptor_load {{.*}} {ttig.desc_padding = 1 : i32}
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.return
  tt.func @if_divergent_last_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16, #dot_a>) {
      // COM: candidate 1 -- non-unit last stride.
      %d1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c2_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d1 : !tt.tensordesc<64x32xf16, #dot_a>
    } else {
      // COM: candidate 0 -- unit last stride.
      %d2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }

  // COM: A single candidate with a non-unit last stride, on both load and store.
  // COM: Everything else (aligned base, pitch, rank 2) satisfies the other gates.
  // CHECK-LABEL: tt.func @single_nonunit_last_stride
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.descriptor_load {{.*}} {ttig.desc_padding = 1 : i32}
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.descriptor_store {{.*}} {ttig.desc_padding = 1 : i32}
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.return
  tt.func @single_nonunit_last_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c2_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %ld : !tt.tensordesc<64x32xf16, #dot_a>, tensor<64x32xf16, #dot_a>
    tt.return
  }

  // COM: Positive control: the same scf.if with a unit last stride on both
  // COM: candidates is stamped, so the negative cases above are not passing
  // COM: merely because some other gate rejects this setup.
  // CHECK-LABEL: tt.func @if_uniform_unit_last_stride
  // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
  // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
  tt.func @if_uniform_unit_last_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16, #dot_a>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d1 : !tt.tensordesc<64x32xf16, #dot_a>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %ld : !tt.tensordesc<64x32xf16, #dot_a>, tensor<64x32xf16, #dot_a>
    tt.return
  }
}
