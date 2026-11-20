// RUN: triton-opt %s -split-input-file -verify-diagnostics --tritonintelgpu-lower-to-2d-block-load

// COM: LowerTo2DBlockLoad rejects descriptor loads whose padding cannot be decided (#8102), as
// COM: DescriptorLoadOpConversion does in ../descriptor-load-divergent-padding.mlir.

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

// -----

// COM: Divergent twin of @if_consistent_padding in
// COM: LowerTo2DBlockLoad/descriptor-load.mlir: with 2D block IO available, the
// COM: 2D path refuses the scf.if candidates that disagree on padding, and the
// COM: fallback rejects them too.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @if_divergent_padding(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32, %arg4: i64, %cond: i1) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%arg4, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %arg1, [%arg2, %arg3], [%arg4, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    // expected-error @+1 {{descriptor padding is divergent: the operations defining this descriptor disagree}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
