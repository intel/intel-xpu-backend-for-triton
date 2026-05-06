#ifndef TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H

#include "intel/include/Analysis/StrideInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu::intel {

/// Determine whether a load op (`tt.load`, `tt.descriptor_load`, or
/// `tt.descriptor_gather`) inside a loop touches the same cache lines on
/// successive iterations of its enclosing loops.
///
/// The analysis walks outward from each load through its enclosing loops.
/// For each enclosing loop L (innermost first), it appends one bool to the
/// result vector:
///   * Case A — pointer loop-invariant w.r.t. L
///     (`loop.isDefinedOutsideOfLoop(ptr)`). Report reuse.
///   * Case B — some tensor axis has IV-stride 0 across L (partial-axis
///     reuse; e.g. a GEMM A-operand's M axis is held while K advances).
///     Report reuse.
///   * Case C — every axis has positive IV-stride across L (pure
///     streaming). Report no reuse.
///
/// `hasTemporalReuse` is the policy wrapper `any_of(getReuseByLoopDepth)`.
///
/// Examples — load with reuse vs load without reuse:
///
/// GEMM A-operand (reuse). Inside the K-loop, %a_tile reloads the same M rows
/// at different K columns on every iteration; the tile stays on the M axis
/// across iterations. hasTemporalReuse = true.
///
///   scf.for %k = %c0 to %K step %BK iter_args(%a_ptr = %a_ptr_init) {
///     %a_tile = tt.load %a_ptr : tensor<BMxBK x !tt.ptr<f16>>
///     %a_next = tt.addptr %a_ptr, %k_offsets
///     scf.yield %a_next
///   }
///
/// 1-D vector copy (no reuse). %ptr advances through distinct BS-element
/// chunks on every iteration; every axis has positive IV-stride, so the
/// analysis correctly reports no reuse.
///
///   scf.for %i = %c0 to %N step %BS iter_args(%ptr = %ptr_init) {
///     %x = tt.load %ptr : tensor<BS x !tt.ptr<f32>>
///     %next = tt.addptr %ptr, %bs_offsets
///     scf.yield %next
///   }
class TemporalReuseAnalysis {
public:
  TemporalReuseAnalysis(
      ModuleOp module,
      mlir::triton::intel::ModuleStrideAnalysis &strideAnalysis)
      : ctx(module.getContext()), strideAnalysis(strideAnalysis) {}

  /// Structural query: one bool per enclosing loop (innermost first,
  /// outermost last). Empty vector => load is not inside any loop.
  SmallVector<bool> getReuseByLoopDepth(mlir::triton::LoadOp op) const;
  SmallVector<bool>
  getReuseByLoopDepth(mlir::triton::DescriptorLoadOp op) const;
  SmallVector<bool>
  getReuseByLoopDepth(mlir::triton::DescriptorGatherOp op) const;

  /// Policy wrapper: true iff any enclosing loop level has temporal reuse.
  bool hasTemporalReuse(mlir::triton::LoadOp op) const;
  bool hasTemporalReuse(mlir::triton::DescriptorLoadOp op) const;
  bool hasTemporalReuse(mlir::triton::DescriptorGatherOp op) const;

private:
  MLIRContext *ctx;
  mlir::triton::intel::ModuleStrideAnalysis &strideAnalysis;

  /// Shared implementation: classify every enclosing loop of `op` given the
  /// set of operands that contribute to the effective address of the load
  /// (e.g. the pointer for `tt.load`; the descriptor + all scalar/tensor
  /// index operands for `tt.descriptor_load` and `tt.descriptor_gather`).
  /// A load has reuse at a given loop level iff every address-determining
  /// operand is either loop-invariant or holds at least one tensor axis
  /// fixed across iterations; conversely, if *any* operand advances along
  /// every one of its axes, the effective address streams and the analysis
  /// reports no reuse.
  SmallVector<bool> classify(Operation *op, ValueRange operands) const;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H
