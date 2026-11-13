// RUN: triton-opt %s -split-input-file -verify-diagnostics --tritonintelgpu-lower-to-2d-block-load

// COM: === Issue #8102: divergent descriptor padding must never reach LLVM ===
// COM:
// COM: On the descriptor-native route the padding decision has to be a compile-time
// COM: constant, and it is carried on the load by `ttig.desc_padding`. When the
// COM: load's provenance has several `tt.make_tensor_descriptor` candidates whose
// COM: `padding` disagrees, `DescriptorDefinitions::consistentPadding()` returns
// COM: nullopt and every producer of that attribute bails. Before the fix the LLVM
// COM: lowering read the *absence* of `ttig.desc_padding` as PAD_ZERO, so a branch
// COM: that asked for a NaN fill silently got zeros.
// COM:
// COM: In the normal pipeline such loads are now expanded to pointers long before
// COM: TTGIR exists, so these cases are unreachable there. This file is the
// COM: backstop for everything that bypasses that expansion -- hand-written TTGIR,
// COM: future passes that reintroduce a divergent merge, or a regression in the
// COM: expansion's legality predicate. Reaching LowerTo2DBlockLoad, which turns
// COM: every descriptor load into a 2D block load or a tt.load, with an
// COM: undecidable padding must be a hard error, not a silent PAD_ZERO.
// COM:
// COM: Every case here fans the provenance out with `arith.select`, deliberately
// COM: NOT `scf.if`. A region-free shape cannot be perturbed by the transient
// COM: empty-region window that issue #8167 is about, so the diagnostics stay
// COM: deterministic and a failure here can only mean the padding check changed.
// COM:
// COM: The shape does not affect these diagnostics: all three `emitError`s fire
// COM: before the boundary-check classification is reached. The [5,5] shape
// COM: (not divisible by the 4x4 block) only matters if the checks are removed,
// COM: in which case the lowering succeeds, emits a predicated load, and the fill
// COM: value it picks is what reaches the masked-off lanes.
// COM:
// COM: Every case is a fix witness: without the fix the lowering emits no padding
// COM: diagnostic at all, so each fails as `expected error "..." was not produced`
// COM: and the masked-off lanes quietly take whatever the pre-fix precedence gave
// COM: them -- the stamped attribute, else the traced padding, else PAD_ZERO.
// COM:
// COM: `-verify-diagnostics` runs in its default strict mode, where an unannotated
// COM: diagnostic is an error, so a duplicate diagnostic fails this file rather
// COM: than passing it quietly.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

// COM: Case 1 -- divergent provenance, NO `ttig.desc_padding` attribute. This is
// COM: what MaterializeBlockPointer leaves for an undecidable padding: it stamps
// COM: nothing, and "nothing" is what the lowering used to turn into PAD_ZERO.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @divergent_padding_no_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <4x4xf32>
    %d1 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %desc = arith.select %cond, %d0, %d1 : !tt.tensordesc<4x4xf32>
    // expected-error @+1 {{descriptor padding is divergent: the operations defining this descriptor disagree}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %0 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

// COM: Case 2 -- divergent provenance WITH a `ttig.desc_padding` attribute. The
// COM: attribute cannot be trusted here no matter what it says: the provenance is
// COM: undecidable, so *some* runtime path disagrees with any single constant. The
// COM: value chosen below (PAD_NAN) is the "lucky" one for one branch and wrong for
// COM: the other, which is why the check must be on the divergence and not on
// COM: whether an attribute happens to be present. Without this case a fix that
// COM: only checked `!hasAttr(ttig.desc_padding)` would look complete.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @divergent_padding_with_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <4x4xf32>
    %d1 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %desc = arith.select %cond, %d0, %d1 : !tt.tensordesc<4x4xf32>
    // expected-error @+1 {{descriptor padding is divergent: the operations defining this descriptor disagree}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %0 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

// COM: Case 3 -- CONSISTENT provenance (both candidates request PAD_NAN) but a
// COM: `ttig.desc_padding` attribute that CONTRADICTS it (PAD_ZERO). The two
// COM: `tt.make_tensor_descriptor` ops are separate ops whatever their operands:
// COM: this RUN line has no CSE, so the trace sees two candidates that agree.
// COM: The distinct base pointers are not needed for that; using `%arg0` for both
// COM: also passes. They only make the two producers visibly independent.
// COM:
// COM: This case has nothing to do with divergence; it guards the other failure
// COM: mode of the same invariant. A stale or mis-stamped attribute is just as
// COM: capable of turning a NaN fill into zeros as a missing one, and the attribute
// COM: is the only thing the lowering actually reads. If the lowering is ever
// COM: changed to consult the provenance directly and ignore the attribute, this is
// COM: the case that should start failing and force that decision to be explicit.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @consistent_padding_mismatched_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %d1 = tt.make_tensor_descriptor %arg1, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %desc = arith.select %cond, %d0, %d1 : !tt.tensordesc<4x4xf32>
    // expected-error @+1 {{'ttig.desc_padding' disagrees with the padding of the operations defining this descriptor}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %0 : tensor<4x4xf32, #blocked>
  }
}

// -----

// COM: Consistent provenance, but nothing stamped the attribute. MaterializeBlockPointer
// COM: stamps exactly when `consistentPadding()` yields a value, so this combination
// COM: cannot come out of that pass. Erroring keeps `ttig.desc_padding` the single
// COM: source of the fill -- the trace only validates it -- rather than re-deriving a
// COM: padding no pass vouched for. @consistent_padding_mismatched_attr above is the
// COM: *wrong*-attribute twin; here the value is simply absent.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @consistent_padding_missing_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %d1 = tt.make_tensor_descriptor %arg1, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <4x4xf32>
    %desc = arith.select %cond, %d0, %d1 : !tt.tensordesc<4x4xf32>
    // expected-error @+1 {{'ttig.desc_padding' is missing but the operations defining this descriptor agree on a padding mode}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %0 : tensor<4x4xf32, #blocked>
  }
}
