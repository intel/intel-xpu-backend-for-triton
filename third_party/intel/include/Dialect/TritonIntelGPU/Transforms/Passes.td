//===-- Passes.td - TritonIntelGPU pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_GPU_PASSES
#define TRITON_INTEL_GPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonIntelGPUAccelerateMatmul
    : Pass<"tritonintelgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Intel accelerate matmul";

  let description = [{
    Optimize the input/output layout of the `tl.dot` operation to make them
    compatible with the Intel DPAS instruction requirements.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonIntelGPUFoldFpToFp
    : Pass<"tritonintelgpu-fold-fp-to-fp", "mlir::ModuleOp"> {
  let summary = "Fold redundant tt.fp_to_fp cast chains";

  let description = [{
    Fold an `A -> B -> C` chain of `tt.fp_to_fp` casts into a single `A -> C`
    cast when the outer cast `B -> C` is lossless (`B` is a subset of `C`),
    dropping the redundant narrow intermediate `B`. Losslessness is decided from
    `llvm::APFloat` semantics (mantissa precision + exponent range), so type
    width alone is never trusted.

    Two regimes:
      * Inner cast `A -> B` is itself lossless (a pure widen, e.g.
        `f8E5M2 -> f16 -> f32`). Dropping `B` preserves the result exactly, so
        the fold runs unconditionally (strict-FP legal).
      * Inner cast `A -> B` is lossy (a real downcast, e.g. user-inserted
        `f32 -> f8 -> f16`). Dropping `B` changes the numerical result, so the
        fold runs only under fast-math.

    Fast-math is read from the `ttig.fast_math` module attribute (set from the
    `TRITON_INTEL_FAST_MATH` environment variable by the annotate-module pass),
    not a pass option.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonIntelGPUStageLargeFMADotsViaSLM
    : Pass<"tritonintelgpu-stage-large-fma-dots-via-slm", "mlir::ModuleOp"> {
  let summary =
      "Stage large blocked-encoding dot operands via SLM and tile K";

  let description = [{
    On non-DPAS Intel GPUs, large `tt.dot` operations with `BlockedEncoding`
    cause per-thread scratch space (PTSS) overflow because the FMA lowering
    fully unrolls the K dimension. This pass operates at the
    `ttg.convert_layout` boundary (or reuses an existing `ttg.local_load`
    from the software pipeliner): full-shape operands are staged into SLM
    via `ttg.local_alloc`, then the dot is unrolled into N concrete
    `ttg.memdesc_subslice` + `ttg.local_load` + `tt.dot` chunks, each with
    the K-dim narrowed to `K_TILE`.

    Operating at `ttg.convert_layout` (rather than rewriting at the load)
    preserves global-memory coalescing and composes with software
    pipelining (which already stages SLM).

    Pass is a no-op on DPAS-capable hardware (PVC, BMG, ARL-H Xe2).
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect"
  ];
}

def TritonIntelGPUOptimizeDotOperands
    : Pass<"tritonintelgpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "Intel optimize dot operands";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands to
    promote the use of hardware-accelerated operations.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonIntelGPUHoistLayoutConversions
    : Pass<"tritonintelgpu-hoist-layout-conversions", "mlir::ModuleOp"> {
  let summary = "Hoist layout conversions out of loops";
  let description = [{
    Hoist `ttg.convert_layout` operations out of `scf.for` loops when the
    destination encoding is `DotOperandEncodingAttr` and the source operand is
    defined outside the loop. This keeps loop-invariant dot operand layout
    conversions in registers, avoiding redundant conversion on every iteration.

    A hoist that would raise the loop body's projected live-in pressure past
    80% of the per-lane GRF budget for the given GRF mode is rejected. The
    projection models a hoist as a *substitution* of the loop body's live-in
    set rather than an addition: the conversion's result starts crossing the
    loop, and when no operation remaining inside the loop reads the
    conversion's source, that source stops crossing it and exactly the bytes it
    contributed to that live-in figure are credited back (so a rematerializable
    source, which contributed none, earns no credit). This matters because a
    hoist whose source layout is
    per-lane fatter than the dot-operand layout it feeds (typical for an
    oversized blocked layout on a narrow tensor) *lowers* live-in pressure,
    and an additive-only estimate would reject it.

    Only a hoist that *spends* budget is subject to the budget. When the
    substitution comes out at zero or negative, the loop is left exactly as far
    over (or under) budget as it already was, so the threshold has nothing to
    say: rejecting such a hoist buys back no register and only forgoes the
    loop-invariant conversion, irrevocably (it stamps `tt.no_licm`). Measured on
    `_attn_bwd` at HEAD_DIM=128, all eight loop-invariant dot-operand
    conversions are break-even substitutions, and hoisting all eight leaves
    every loop body's live-in pressure (292/260/452/516 bytes/lane) and the
    whole-function peak (2048 bytes/lane) bit-for-bit unchanged while dropping
    one loop's peak by 64 bytes/lane. Gating them on the unchanged live-in
    figure rejected six and cost 21.7% geomean on flash-attn-bwd.

    A source is only credited when the hoist really retires it: nothing else
    inside the loop may read it, and nothing after the loop may either (a source
    read below the loop occupies a register for the loop's whole duration
    wherever the conversion sits, so moving the conversion frees nothing). A
    source shared by several conversions in the same loop is therefore credited
    only once, to whichever conversion is the last to leave the loop.

    Both halves of that question are asked of the IR as it stands, so a
    conversion this pass has already hoisted counts as moved. Two decisions
    follow from that:

    * Candidates belonging to one loop are decided as a set, cheapest (most
      pressure-reducing) first, rather than in the order they appear in the loop
      body. Because a hoist can lower the projection as well as raise it,
      deciding in program order would make an early rejection depend on credit
      that a later candidate had not yet contributed.
    * The loops themselves are decided last to first. A hoist only ever moves a
      conversion earlier in its block, so once every loop below a given loop has
      been decided, no later hoist can add or remove a use of a source below it.
      Deciding first-to-last instead denied the credit to the earlier of two
      loops sharing a source, purely because the later loop's conversion had not
      moved out from under it yet.

    Both orderings make the pass's output a function of the loops' register
    pressure rather than of their syntax.

    Non-goals of the pressure model, all deliberate:

    * **Transient overlap at the hoist site.** The projection is about the
      loop body's live-in pressure. When the source and the hoisted result are
      the same per-lane size, hoisting leaves live-in unchanged but moves the
      moment at which both are simultaneously live out of the loop and into
      the straight-line code before it, which can *raise* whole-function peak
      pressure even as loop peak falls. That program point is not measured, so
      such a hoist is judged solely on the (unchanged) loop live-in figure --
      and, per the rule above, is therefore taken. It does not bite on
      `_attn_bwd`, whose whole-function peak is unchanged by all eight hoists,
      but a kernel where it does would be hoisted anyway.
    * **A block-argument source is not freed for earlier loops.** A conversion
      whose source has no defining op is hoisted to immediately before its own
      loop, which is still *after* any earlier loop sharing that source. The
      earlier loop then correctly earns no credit (the source really does still
      cross it), so the last-to-first ordering only helps sources with a
      defining op to hoist above -- which is the common case, the source being a
      load.
    * **Groups sharing a source are not evaluated jointly.** Since the credit
      only lands on the last conversion to leave, hoisting the first `k-1` of
      `k` conversions that share a source projects the arrivals with none of the
      departure. With three or more sharers under a tight budget the pass can
      therefore stop partway through a group whose completed net effect would
      have fit the budget. The error is in the conservative direction (too few
      hoists, never too many).
    * **Loop trip count and conversion cost.** The saved per-iteration
      conversion work is not weighed against the pressure change, and the trip
      count is not consulted.
    * **Pre-existing loop pressure.** `liveInPressure` counts only values used
      inside the loop body, so a value that merely spans the loop to feed code
      below it is invisible to the baseline the threshold is compared against.
      The gate is a check on what a hoist *adds*, not a claim about the loop's
      absolute occupancy.

    A rejected conversion is marked `tt.no_licm` so that later generic LICM
    does not hoist it anyway.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    Option<"grfMode", "grf-mode", "std::string", /*default*/"\"default\"",
           "GRF mode (default, 128, 256, or auto)">,
  ];
}

def TritonIntelGPUPipeline : Pass<"tritonintelgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Pipeline loops";

  let description = [{
    Apply software pipelinining to loops containing `tt.dot` operations.
    The pass supports prefetching `tt.dot` operands. The `num-stages` argument controls
    the prefetching and distance (i.e. the number of iterations to prefetch in advance).
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::spirv::SPIRVDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"useBarrier", "use-barrier", "bool", /*default*/"false",
           "generate barriers">,
  ];
}

def TritonIntelGPURemoveLayoutConversions : Pass<"tritonintelgpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
    This is a customized remove layout conversion for Intel GPU arch.
    Different GPUs characteristics make it profitable for Intel HW to load the
    operands of a `tt.dot` operation into registers.
    Therefore given the following example:

    ```mlir
    #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
    #dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.descriptor_load %desc_a[%off_m, %off_k] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    %29 = tt.descriptor_load %desc_b[%off_k, %off_n] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked1>
    %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>>
    %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
    ```

    After this pass, the convert layout ops is removed which deviates from the
    common TTGIR remove layout conversion.
    Like this:

    ```mlir
    #dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.descriptor_load %desc_a[%off_m, %off_k] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>>
    %29 = tt.descriptor_load %desc_b[%off_k, %off_n] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
    ```

    On NVidia GPUs it is profitable to load the operands of a tt.dot operation
    into shared local memory (therefore the layout conversion operations are
    necessary). On Intel GPUs loading into SLM is not profitable because it
    would require synchronization operations that are expensive. Therefore it
    is better to load the operands directly into registers and incur the cost
    of duplicating the load, because the HW can combine redundant memory
    accesses in the IO buffer or cache them.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"maxBackwardRematIterations", "max-backward-remat-iterations",
           "unsigned", /*default*/"1",
           "Maximum number of backward rematerialization iterations">,
  ];
}

def TritonIntelGPUReduceDataDuplication: Pass<"tritonintelgpu-reduce-data-duplication", "mlir::ModuleOp"> {
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = [{
    Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors.
    This Intel pass supports the Intel DPAS layout in additional to the upstream Triton pass.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonIntelGPUMaterializeBlockPointer : Pass<"tritonintelgpu-materialize-block-pointer", "mlir::ModuleOp"> {
  let summary = "annotate load operations with information required to exploit 2D block HW instructions";

  let description = [{
    This pass annotates load operations using a block pointer with information required to exploit 2D
    block HW instructions during lowering (e.g. whether the memory access pattern is row or column major).
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonIntelGPUWidenLoadStoreEncoding : Pass<"tritonintelgpu-widen-load-store-encoding", "mlir::ModuleOp"> {
  let summary = "Widen a load/store's blocked encoding from 128b to 256b per thread";

  let description = [{
    On Xe3P (gated by the `ttig.support_256b_load_store` module attribute), this pass
    rewrites `tt.load` and `tt.store` operations whose tensor uses a `BlockedEncodingAttr`
    with a 128-bit-per-thread contiguous tile into a 256-bit-per-thread contiguous tile,
    when pointer alignment, contiguity, and tensor shape allow it. Descriptor-based
    loads/stores (`tt.descriptor_load`, `tt.descriptor_store`) and atomic ops
    (`tt.atomic_rmw`) are out of scope. The pass relies on the downstream
    `tritonintelgpu-remove-layout-conversions` pass to absorb any `ttg.convert_layout`
    ops introduced by the rewrite.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];
}

def TritonIntelGPUOptimizeReductionLocality
    : Pass<"tritonintelgpu-optimize-reduction-locality", "mlir::ModuleOp"> {
  let summary = "Minimize number of reductions within sub-groups";

  let description = [{
    This pass performs layout conversions so `tt.reduce` operations resulting in
    sub-group reductions are converted to `tt.reshape`, `tt.reduce`, and
    `ttg.convert_layout` operations, e.g.:
    ```mlir
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2]}>
tt.func @test(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) : (tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
}
    ```
    Is converted to:
    ```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 8, 2, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4, 5, 6]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [0, 1, 2, 3]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
tt.func @test(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
  %0 = tt.reshape %arg0 allow_reorder efficient_layout : tensor<16x32xf32, #mma> -> tensor<16x8x2x2x1x1x1xf32, #blocked>
  %1 = "tt.reduce"(%0) <{axis = 2 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x8x2x2x1x1x1xf32, #blocked>) -> tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
  %2 = "tt.reduce"(%1) <{axis = 4 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #blocked}>>) -> tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
  %3 = ttg.convert_layout %2 : tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<16x8x2x1x1xf32, #blocked1>
  %4 = tt.reshape %3 allow_reorder efficient_layout : tensor<16x8x2x1x1xf32, #blocked1> -> tensor<16x16x1x1xf32, #blocked2>
  %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x16x1x1xf32, #blocked2>) -> tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
  %6 = "tt.reduce"(%5) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>) -> tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #blocked2}>}>>
  %7 = tt.reshape %6 allow_reorder efficient_layout : tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #blocked2}>}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
  %8 = ttg.convert_layout %7 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  tt.return %8 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
}
    ```
    The `tt.reshape` operation is a NOP so that the following `tt.reduce`
    operation performs a reduction within the work-item. Then,
    `ttg.convert_layout` performs the actual data movement to prevent
    within the sub-group reductions so that the following `tt.reduce` performs
    a reduction within the work-item again. Finally, we convert back to the
    original type. Note the order of the operations to go back to the original
    type is important: reshape to original shape and set an anchor for the
    layout conversion removal pass, and convert to original layout.

    Note this pass only supports `ttig.dpas` input layouts at the
    moment, but it should be easily extended.

    See pass implementation for more detailed implementation documentation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonIntelGPURewriteStackPtr
    : Pass<"tritonintelgpu-rewrite-stack-ptr", "mlir::ModuleOp"> {
  let summary = "rewrite the getStackPointer for Intel by addressofOp replacement";

  let description = [{
    This pass rewrites the addressOfOps referring to the `global_smem` symbol,
    depending on how shared memory was allocated by the TritonIntelGPU to LLVM
    conversion:

    - statically (internal linkage): nothing to do, the address of `global_smem`
      is a valid shared memory base.
    - dynamically (external linkage): a shared memory pointer argument is
      appended to every kernel and the addressOfOps are replaced by it.

    In both cases, kernels not using shared memory (`ttg.shared` is 0) get a
    PoisonOp instead, as a zero sized array cannot be materialized in SPIR-V.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect", "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonIntelGPUReduceVariableLiveness
    : Pass<"tritonintelgpu-reduce-variable-liveness", "mlir::ModuleOp"> {
  let summary = "Attempt to reduce the variable liveness";

  let description = [{
    This pass attempts to reduce the variable liveness
    by reducing the distance between loads and usage.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonIntelGPUCodeSinking
    : Pass<"tritonintelgpu-code-sinking", "mlir::ModuleOp"> {
  let summary = "Sink pure ops past a region-bearing op to their first use after the region";

  let description = [{
    Moves a pure (read-only or effect-free, non-volatile) operation defined
    before a region-bearing op (e.g. `scf.for`, `scf.while`, `scf.if`), whose
    results are never used inside the region, down to just before its first use
    after the region. This shortens the live range of a value that would
    otherwise be live across the entire region, reducing register pressure that
    competes with DPAS accumulators and A/B operand tiles. Loads (e.g. the
    SwiGLU bias loads of #7250) are the motivating case, but the transform
    applies to any pure op.

    An op is sunk only when: it is defined in the same block as and before the
    region-bearing op; it has no use inside the region (including uses that flow
    through `scf.yield` or as operands to the barrier op itself); every
    remaining use is strictly after the region; and there is no write side
    effect (including writes inside the region) on the path between the op and
    the sink point.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect"];
}

def TritonIntelGPUAnnotateCacheControl
    : Pass<"tritonintelgpu-annotate-cache-control", "mlir::ModuleOp"> {
  let summary = "Annotate tt.load ops with .cg cache control unless the load's encoding indicates cross-subgroup L1 reuse";
  let description = [{
    Sets `.cg` (L1UC_L3C) cache control on `tt.load` operations whose result
    does not participate in a matrix multiply. GEMM operand loads benefit from
    L1 reuse and are left with the default cache modifier; streaming /
    elementwise loads bypass L1 to avoid thrashing. User-specified cache
    modifiers are never overridden.

    `.cg` on loads is always coherence-safe: it reads from the shared coherent
    L3 rather than the incoherent per-core L1. The filters below are
    cost-model heuristics that preserve L1 locality for loads likely to benefit
    from L1 reuse (e.g. matmul operand loads in complex kernels).

    The pass runs **after** `remove_layout_conversions`, so load-result
    encodings are stable. The heuristic inspects each `tt.load`'s result
    encoding: if it is a `DotOperandEncodingAttr` whose parent implements
    `MmaEncodingTrait` (DPAS / Subgroup2DBlockEncoding), annotation is
    suppressed — the HW access pattern already yields cross-subgroup L1 reuse
    because the K dimension is broadcast across warps. Nested
    `SliceEncodingAttr` layers are unwrapped before the dot-operand check.
    The load is also suppressed when its result feeds the `aScale`/`bScale`
    operand of a `tt.dot_scaled` — a single-hop use-site check suffices
    because AccelerateMatmul materialises scale encodings immediately before
    the consumer, with no `scf.for` / `convert_layout` in between.

    To avoid unnecessary L1 bypass, loads are additionally skipped when their
    pointer root-traces to a kernel argument that falls into any of the
    following excluded cases (computed per `tt.FuncOp`):

      1. Read-write args: arguments that are both loaded and stored within the
         same function (producer/consumer scratch buffers where L1 reuse is
         likely beneficial).
      2. Atomic kernels: if the function contains any `tt.atomic_rmw` or
         `tt.atomic_cas`, every pointer-typed entry-block argument is
         conservatively excluded, even if only some of them are loaded.
         Atomic kernels often mix matmul operand loads with streaming loads;
         excluding all args avoids regressing matmul operand L1 reuse.
      3. Unresolved stores: if any `tt.store` has a pointer SSA chain that
         cannot be resolved to known roots, every pointer-typed entry-block
         argument is excluded (we cannot prove which args are written).
      4. Unresolved load pointer: if a `tt.load`'s own pointer SSA chain hits
         an op the pass does not whitelist, the load is skipped.
      5. Forward flow into excluded store: if the loaded value transitively
         feeds a `tt.store` whose destination resolves to an excluded arg, the
         load is skipped even if its own pointer is read-only. This is a
         defensive filter that may be redundant.

    Stores are intentionally left untouched: unlike loads, annotating stores
    with `.cg` on cross-kernel producer/consumer buffers (e.g. layer-norm
    backward partial-sum scratch) introduces data races — a true coherence
    concern.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect"];
}

def TritonIntelGPULowerTo2DBlockLoad
    : Pass<"tritonintelgpu-lower-to-2d-block-load", "mlir::ModuleOp"> {
  let summary = "Lower eligible loads to ttig.2d_block_load ops";

  let description = [{
    Converts eligible `tt.load` and `tt.descriptor_load` operations into
    `ttig.2d_block_load` operations with decomposed surface parameters. This
    separates the decision to use 2D block I/O from the LLVM lowering mechanics.
    Loads that cannot be converted remain unchanged and are handled by existing
    LLVM lowering patterns.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect"
  ];
}

def TritonIntelGPUCanonicalizePointers
    : Pass<"tritonintelgpu-canonicalize-pointers", "mlir::ModuleOp"> {
  let summary = "Canonicalize pointers: decompose tensor-pointer arithmetic "
                "into scalar-base + tensor-offset form";
  let description = [{
    Traces every tt.ptr-typed function argument through its transitive uses and
    decomposes tensor pointer arithmetic into a FatPtr{scalar_base,
    tensor_offset} pair. Uniform (scalar-extractable) parts of each offset are
    hoisted onto the scalar base pointer; only the non-uniform residual remains
    as the tensor offset. At load/store sites the pass materializes
    splat(scalar_base) + tensor_offset.

    This reduces register pressure for pointer-heavy kernels by replacing
    loop-carried tensor<Nx!tt.ptr> iter_args with scalar !tt.ptr iter_args.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect"
  ];
  let options = [
    Option<"enableLargeTensorPtrCanon", "enable-large-tensor-ptr-canon",
           "bool", /*default=*/"false",
           "Enable canonicalization for pointers to tensors larger than 2GB">
  ];
}

def TritonIntelGPULoopDistribute
    : Pass<"tritonintelgpu-loop-distribute", "mlir::ModuleOp"> {
  let summary = "Distribute dot operations in loops into separate loops";

  let description = [{
    Split an `scf.for` loop containing multiple `tt.dot` operations into
    separate loops, each containing a single `tt.dot`. This reduces register
    pressure (fewer live accumulators) and descriptor pressure (fewer active
    2D block I/O descriptors per loop), at the cost of re-loading shared
    operands.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // TRITON_INTEL_GPU_PASSES
