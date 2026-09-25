// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: scf.for with pass-through yield (descriptor unchanged across iterations).
// COM: findDescriptorDefinitions should resolve to the unique MakeTensorDescOp.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_passthrough_yield
  tt.func @for_passthrough_yield(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %iter_desc : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}

// -----

// COM: scf.for where the yield provides a different MakeTensorDescOp with compatible
// COM: alignment properties. All candidates satisfy constraints, so block_io is set.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_multiple_compatible_descs
  tt.func @for_multiple_compatible_descs(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %desc2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc1) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %desc2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}

// -----

// COM: scf.for where the yield provides a MakeTensorDescOp with incompatible
// COM: pitch (not divisible by 128/elementWidth). block_io should NOT be set.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_incompatible_pitch
  tt.func @for_incompatible_pitch(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch_good: i64 {tt.divisibility = 16 : i32}, %pitch_bad: i64 {tt.divisibility = 3 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch_good, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %desc2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch_bad, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc1) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %desc2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}

// -----

// COM: Positive control for scf.if: two candidates that agree on padding and
// COM: reuse the same shape/stride SSA values. consistentShape() compares
// COM: operand ranges by SSA identity, so the constants must be defined once
// COM: outside the scf.if -- separate `arith.constant 64` ops would disagree.
// COM: Without this case the negative scf.if cases below would pass even if the
// COM: worklist failed to trace through scf.if at all.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @if_consistent_descs
  tt.func @if_consistent_descs(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
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
    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

// COM: scf.if whose candidates disagree on padding: PAD_ZERO (the default) from
// COM: the then branch, PAD_NAN (`padding = 2 : i32`) from the else branch.
// COM: consistentPadding() returns nullopt and bails before either stamp, so the
// COM: load keeps no attribute dictionary at all.
// COM:
// COM: Stamping nothing is the CORRECT and COMPLETE behaviour, but for two
// COM: different reasons depending on how this IR was reached:
// COM:
// COM: (a) In the normal pipeline this TTGIR is unreachable. A descriptor load
// COM:     whose provenance disagrees on padding is routed to the pointer
// COM:     expansion (--triton-intel-rewrite-tensor-descriptor-to-pointer), which
// COM:     models padding as a runtime i1 and selects between a NaN splat and a
// COM:     zero splat -- see @if_divergent_padding in
// COM:     test/Triton/Intel/rewrite-tensor-descriptor-to-pointer.mlir. So by the
// COM:     time TTGIR exists there is no divergent descriptor left, and refusing
// COM:     to stamp a padding here costs nothing.
// COM:
// COM: (b) For standalone hand-written TTGIR like this fixture, which never went
// COM:     through that expansion, refusing is still correct -- taking the 2D block
// COM:     path with a guessed padding would be strictly worse -- and it is no
// COM:     longer the last line of defence: such a load is now rejected outright by
// COM:     the LLVM lowering, which errors instead of silently defaulting to
// COM:     PAD_ZERO. See test/TritonIntelGPU/descriptor-load-divergent-padding.mlir.
// COM:
// COM: Issue #8102 is the silent-PAD_ZERO degradation that both of those close.
// COM: This case asserts only the attribute-stamping half: no ttig.desc_padding,
// COM: therefore no ttig.block_io.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @if_divergent_padding
  tt.func @if_divergent_padding(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16, #dot_a>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d1 : !tt.tensordesc<64x32xf16, #dot_a>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    // COM: The ` :` immediately after the indices proves the load carries no
    // COM: attribute dictionary, hence neither ttig.desc_padding nor
    // COM: ttig.block_io.
    // CHECK: tt.descriptor_load %{{[0-9]+}}[%c0_i32, %c0_i32] :
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

// COM: scf.if whose candidates agree on padding but use different shape
// COM: operands. Both shapes are independently 2D-block-legal (last extents 32
// COM: and 64 are both divisible by ceil(32/16) = 2 for f16), so block_io is
// COM: absent because consistentShape() returned nullopt, not because of an
// COM: unrelated base-width failure.
// COM:
// COM: This is the case that pins the ordering constraint in visitDescriptor:
// COM: the padding stamp must precede the shape bail. Hoisting consistentShape()
// COM: above it would drop ttig.desc_padding here and silently turn a PAD_NAN
// COM: descriptor's out-of-bounds fill into zeros.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @if_divergent_shape
  tt.func @if_divergent_shape(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16, #dot_a>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d1 : !tt.tensordesc<64x32xf16, #dot_a>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %d2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    // COM: Pinning the whole attribute dictionary is what proves ttig.block_io
    // COM: is absent -- it sorts before ttig.desc_padding, so a CHECK-NOT
    // COM: anchored after the desc_padding match would not see it.
    // CHECK: tt.descriptor_load {{.*}} {ttig.desc_padding = 1 : i32} :
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

// COM: Rank-0 descriptor. `rank` is unsigned, so before the `rank < 2` guard
// COM: this fell through to `rank - 1` / `rank - 2` and indexed the shape and
// COM: stride operand ranges far out of bounds. Verified against ac39409b1:
// COM: triton-opt aborts (exit 134) on OperandRange::operator[]'s
// COM: `Index < size()` assertion, so under NDEBUG this is an out-of-bounds
// COM: read, not merely an unsigned wrap. A 0-D descriptor is representable:
// COM: the type is a dimension list plus a scalar element type, and an empty
// COM: dimension list parses.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @rank0_desc
  tt.func @rank0_desc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %desc = tt.make_tensor_descriptor %arg0, [], [] : !tt.ptr<f32>, !tt.tensordesc<f32>
    // CHECK: tt.descriptor_load %{{[0-9]+}}[] {ttig.desc_padding = 1 : i32} :
    %ld = tt.descriptor_load %desc[] : !tt.tensordesc<f32> -> tensor<f32>
    tt.return
  }
}

// -----

// COM: A loop result is its tied init when the loop runs zero times (6bd0d53ae).
// COM: %r is %dZ (PAD_ZERO) on a zero-trip loop and %dN (PAD_NAN) otherwise, so its
// COM: provenance is divergent and consistentPadding() stamps nothing.
// COM:
// COM: Measured on a pre-branch binary: the provenance followed only the yield, so
// COM: the load was stamped `{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}`,
// COM: i.e. a zero-trip loop would fill out-of-bounds elements with NaN instead of 0.
// COM:
// COM: As in @if_divergent_padding, the ` :` immediately after the indices on the
// COM: same line proves the load carries no attribute dictionary, hence neither
// COM: ttig.desc_padding nor ttig.block_io. (A CHECK-NOT after a
// COM: `CHECK: tt.descriptor_load` would start on the next line and miss it.)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_zero_trip_divergent_padding
  tt.func @for_zero_trip_divergent_padding(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %dZ = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 1 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %r = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%x = %dZ) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      %dN = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.yield %dN : !tt.tensordesc<64x32xf16, #dot_a>
    }
    // CHECK: tt.descriptor_load %{{[0-9]+}}[%c0_i32, %c0_i32] :
    %ld = tt.descriptor_load %r[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

// COM: An scf.while result is the matching scf.condition operand, not the
// COM: after-region yield (0fa440764). %r is always %dN (PAD_NAN); the PAD_ZERO %dZ
// COM: only re-enters the before region, so the load must be stamped PAD_NAN.
// COM:
// COM: Measured on a pre-branch binary: the provenance followed the after-region
// COM: yield (%dZ), so the load was stamped `ttig.desc_padding = 1 : i32` and a
// COM: PAD_NAN descriptor would have been filled with zeros.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @while_condition_padding
  tt.func @while_condition_padding(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %c: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %dZ = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 1 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %r = scf.while (%x = %dZ) : (!tt.tensordesc<64x32xf16, #dot_a>) -> !tt.tensordesc<64x32xf16, #dot_a> {
      %dN = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
      scf.condition(%c) %dN : !tt.tensordesc<64x32xf16, #dot_a>
    } do {
    ^bb0(%y: !tt.tensordesc<64x32xf16, #dot_a>):
      scf.yield %dZ : !tt.tensordesc<64x32xf16, #dot_a>
    }
    // CHECK: tt.descriptor_load %{{[0-9]+}}[%c0_i32, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32} :
    %ld = tt.descriptor_load %r[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}
