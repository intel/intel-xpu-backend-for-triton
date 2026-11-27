#ifndef TRITON_INTEL_ANALYSIS_UTILITY_H
#define TRITON_INTEL_ANALYSIS_UTILITY_H

#include <optional>

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {

struct SubGroupReinterpretPackInfo {
  bool isPack;
  unsigned packedRegisterSize;
};

std::optional<LinearLayout>
getReinterpretCastMapping(MLIRContext *ctx, const LinearLayout &srcLayout,
                          const LinearLayout &dstLayout);

/// Return pack/unpack metadata for a valid sub-group reinterpret mapping.
std::optional<SubGroupReinterpretPackInfo>
getSubGroupReinterpretPackInfo(MLIRContext *ctx,
                               const LinearLayout &conversion);

/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group shuffle.
bool cvtIsSubGroupShuffle(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group transpose through local memory.
bool cvtIsSubGroupTranspose(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether the layout conversion performed by `op` can be performed as a
/// sub-group bitcast shuffle (reinterpret cast).
bool cvtIsSubGroupReinterpret(ConvertLayoutOp op);
/// Return whether `type` is a valid element type for a fast sub-group
/// transpose.
bool isValidElementTypeForSubGroupTranspose(Type type);
/// Return the number of elements per row of the local-memory matrix a sub-group
/// transpose goes through, for a sub-group of `threadsPerWarp` work-items and
/// `bytesPerElement`-byte elements. Rows hold one element per work-item plus
/// padding; the padding keeps lanes reading a column on different banks and,
/// for elements narrower than a dword, keeps every row dword-aligned.
unsigned getSubGroupTransposeRowLength(unsigned threadsPerWarp,
                                       unsigned bytesPerElement);

// The following four queries used to live on core's `ReduceOpHelper`, but the
// Intel backend is their only consumer: core's own reduction lowering computes
// its scratch space from linear layouts instead. They are kept here so the
// shared analysis stays vendor-neutral.
//
// `helper` is passed in (rather than constructed here) so that the caller's
// existing helper - and its diagnostics - are reused unchanged.

/// Return whether the reduction needs no inter-warp communication.
bool isWarpSynchronous(ReduceOpHelper &helper, triton::ReduceOp op);

/// Return the shape of the shared memory space needed for the reduction.
SmallVector<unsigned> getScratchRepShape(ReduceOpHelper &helper,
                                         triton::ReduceOp op);

/// Return the source layout order with the reduction axis moved to the front.
SmallVector<unsigned> getOrderWithAxisAtBeginning(ReduceOpHelper &helper,
                                                  triton::ReduceOp op);

/// Return the size in bytes of the shared memory space needed for the
/// reduction.
unsigned getScratchSizeInBytesOld(ReduceOpHelper &helper, triton::ReduceOp op);

/// A scan over a blocked layout, as core's scan lowering from before
/// triton-lang/triton#12184 measured it. The Intel backend keeps that lowering
/// for blocked layouts, see TritonIntelGPUToLLVM/ScanOpToLLVM.cpp.
struct BlockedScanLayout {
  /// std::nullopt for other layouts and for scans whose axis spans CTAs.
  static std::optional<BlockedScanLayout> get(triton::ScanOp op);

  /// The operand layout without broadcast registers, and the blocked order.
  LinearEncodingAttr encoding;
  SmallVector<unsigned> order;
  SmallVector<int64_t> shape;
  unsigned axis;
  /// Axis elements a thread holds contiguously, and their register distance.
  unsigned elemsPerThread, elemStride;
  /// Lanes along the axis, and their lane id distance.
  unsigned lanes, laneStride;
  unsigned warps;
  /// Repetitions of the CTA tile along the axis, and their register distance
  /// in units of the repetitions along the other dimensions.
  unsigned blocks, blockStride;
  unsigned parallelElemsPerThread, parallelBlocks;
  /// Threads holding other scans, per warp and per CTA.
  unsigned parallelLanesPerWarp, parallelThreads;
  /// Shared memory elements each operand needs when the axis spans warps, and
  /// the bytes all operands need.
  unsigned scratchElems, scratchBytes;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
