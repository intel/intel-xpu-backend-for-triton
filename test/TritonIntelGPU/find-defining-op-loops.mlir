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
// COM: This pins current behaviour, which is NOT correct behaviour. With no
// COM: ttig.desc_padding the generic descriptor lowering in LoadStoreOpToLLVM
// COM: defaults to PAD_ZERO, so at runtime the branch selecting the PAD_NAN
// COM: descriptor still gets a zero out-of-bounds fill. That hole predates this
// COM: test -- refusing the 2D block path is strictly safer than taking it with
// COM: the wrong padding -- and is tracked separately. Do not read this case as
// COM: "divergent padding is handled".
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
