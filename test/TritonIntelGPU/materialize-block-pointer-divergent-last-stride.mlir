// RUN: not --crash triton-opt %s --tritonintelgpu-materialize-block-pointer 2>&1 | FileCheck %s

// COM: visitDescriptor stamps ttig.block_io = "row_major" covering EVERY
// COM: candidate, so it asserts that every candidate has a unit last stride --
// COM: not just candidate 0.
// COM:
// COM: Candidate order matters here. The worklist pushes the then yield and then
// COM: the else yield, and pops from the back, so the ELSE branch's descriptor is
// COM: candidate 0. Giving the else branch stride 1 and the then branch stride 2
// COM: means the narrow candidate-0-only check this replaced would pass. Reversed,
// COM: the old check aborts too and the case shows nothing.
// COM:
// COM: The IR is valid: tt.make_tensor_descriptor carries no unit-stride
// COM: invariant (only Pure + SameVariadicOperandSize) and the descriptor
// COM: load/store verifier checks element type and total element count, nothing
// COM: about stride values. So this documents a deliberate debug-build abort on
// COM: verified IR -- the assert states what the Intel 2D block path requires,
// COM: not what the dialect guarantees. If it ever fires in CI the fix is to
// COM: convert it to a bail-out, not to weaken it.
// COM:
// COM: Assertions-dependent: in a build with NDEBUG the predicate is not
// COM: evaluated, triton-opt exits 0 and `not --crash` reports failure. Every
// COM: documented build path for this project enables assertions -- setup.py
// COM: defaults to TritonRelBuildWithAsserts (-O2 -g, no -DNDEBUG) and
// COM: scripts/compile-triton.sh exports DEBUG=1.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @if_divergent_last_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16, #dot_a>) {
      // COM: candidate 1 -- non-unit last stride, only the widened check sees it.
      %d1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c2_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d1 : !tt.tensordesc<64x32xf16, #dot_a>
    } else {
      // COM: candidate 0 -- unit last stride.
      %d2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    // CHECK: Tensor descriptor must have stride=1 in last dimension
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}
