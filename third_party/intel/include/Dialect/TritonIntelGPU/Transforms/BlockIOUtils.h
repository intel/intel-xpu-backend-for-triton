#ifndef TRITONINTELGPU_TRANSFORMS_BLOCKIOUTILS_H
#define TRITONINTELGPU_TRANSFORMS_BLOCKIOUTILS_H

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir::triton::intel {
class ModuleStrideAnalysis;
} // namespace mlir::triton::intel

namespace mlir::triton::gpu::intel {

/// Information about a 2D block I/O tile shape, computed from a LinearLayout.
struct BlockIOTileSizeInfo {
  BlockIOTileSizeInfo() = delete;
  BlockIOTileSizeInfo(int tileHeight, int tileWidth, int numElemPerPackedVal,
                      int vBlocks, int rowDim, int colDim, bool transpose,
                      bool vnni,
                      std::optional<SetVector<unsigned>> regPackedBases)
      : tileHeight(tileHeight), tileWidth(tileWidth),
        numElemPerPackedVal(numElemPerPackedVal), vBlocks(vBlocks),
        rowDim(rowDim), colDim(colDim), transpose(transpose), vnni(vnni),
        regPackedBases(regPackedBases) {}
  static BlockIOTileSizeInfo unknown() {
    return {-1, -1, -1, -1, -1, -1, false, false, std::nullopt};
  }

  int tileHeight;
  int tileWidth;
  int numElemPerPackedVal;
  int vBlocks;
  int rowDim;
  int colDim;
  bool transpose;
  bool vnni;
  std::optional<SetVector<unsigned>> regPackedBases;

  bool isValid() const {
    return tileHeight >= 0 && tileWidth >= 0 && numElemPerPackedVal >= 0 &&
           vBlocks >= 0 && rowDim >= 0 && colDim >= 0;
  }
};

BlockIOTileSizeInfo getBlockIOLoadTileSize(const LinearLayout &, unsigned,
                                           unsigned, AxisInfo *, bool);
BlockIOTileSizeInfo getBlockIOPrefetchTileSize(const LinearLayout &, unsigned,
                                               unsigned, AxisInfo *,
                                               bool allow256Bytes);
/// Return the tensor dimension along which consecutive lanes (the fastest-
/// varying lane bit) advance in `ll` -- the dimension the hardware vectorizes
/// for 2D block I/O. This is the dimension getBlockIOTileSize uses to decide
/// block-I/O vs. scatter (block I/O requires it to equal the memory-contiguous
/// dimension). Returns std::nullopt when the first lane basis vector is not a
/// clean single-dimension stride, in which case the layout is not a 2D block
/// I/O candidate.
std::optional<unsigned> getLaneFastChangeDim(const LinearLayout &ll,
                                             MLIRContext *ctx);

/// Get the DPAS operand index from a tensor type's encoding.
/// The encoding must be DPAS or DotOperand-with-DPAS parent.
DpasEncodingAttr::OpIdx getOpIdx(RankedTensorType tensorTy);

/// Get the DPAS layout from a tensor type's encoding.
/// The encoding must be DPAS or DotOperand-with-DPAS parent.
DpasEncodingAttr getDpasLayout(RankedTensorType tensorTy);

/// Compute the shuffle mapping for transposed 2D block loads.
/// Returns failure if the transpose configuration is unsupported.
/// Used by both the TTGIR validation (to reject unsupported configs)
/// and the LLVM lowering (to produce the actual mapping).
FailureOr<LinearLayout> computeTransposeShuffleMapping(
    RankedTensorType tensorType, const LinearLayout &regMapping,
    int64_t numElemsPerLoad, const BlockIOTileSizeInfo &sizeInfo,
    unsigned threadsPerWarp, bool hasDPASOperandType, MLIRContext *ctx);

/// Check whether the packed element size and tile width satisfy the
/// 2D block address payload hardware restrictions.
bool check2DBlockAddressPayloadRestriction(unsigned packedElemSizeInBits,
                                           unsigned tileWidth);

/// Validate that a load with the given encoding and element size can be
/// lowered to 2D block I/O. Checks tile size, HW address restrictions,
/// inner-dim constraints, and transpose shuffle mapping. Returns true if valid.
bool validate2DBlockLoadTile(const LinearLayout &ll, unsigned memContiguousDim,
                             unsigned elemSizeInBits,
                             RankedTensorType tensorType,
                             bool oneMatrixPerLoadForBT = false,
                             AxisInfo *maskAxisInfo = nullptr);

/// Validate that a store with the given encoding and element size can be
/// lowered to 2D block I/O, applying the constraints the store lowering
/// enforces: a valid tile shape, the HW address payload restriction, no
/// transpose, and a single v-block. On success returns true and writes the
/// validated tile geometry (with vBlocks forced to 1) into \p sizeInfoOut; on
/// failure returns false and leaves \p sizeInfoOut unspecified.
///
/// This is the single source of truth for 2D block store eligibility, shared
/// by the store lowering patterns in LoadStoreOpToLLVM.cpp so their tile
/// validation cannot drift.
bool validate2DBlockStoreTile(const LinearLayout &ll, unsigned memContiguousDim,
                              unsigned elemSizeInBits,
                              RankedTensorType tensorType,
                              AxisInfo *maskAxisInfo,
                              BlockIOTileSizeInfo &sizeInfoOut);

/// Determine whether memory layout is row-major from the block_io attribute.
bool isMemoryRowMajor(Operation *op);

/// Check whether a load is eligible for 2D block IO lowering based on
/// attributes and encoding. Callers holding a generic Operation* that has
/// already been confirmed to be tt::LoadOp or tt::DescriptorLoadOp may use
/// this overload directly.
bool isBlockIOEligible(Operation *loadOp, RankedTensorType tensorTy);

/// Type-safe overload restricted to tt::LoadOp and tt::DescriptorLoadOp.
template <typename OpTy,
          std::enable_if_t<llvm::is_one_of<OpTy, triton::LoadOp,
                                           triton::DescriptorLoadOp>::value,
                           bool> = true>
bool isBlockIOEligible(OpTy loadOp, RankedTensorType tensorTy) {
  return isBlockIOEligible(loadOp.getOperation(), tensorTy);
}

/// Estimate the hardware message count for a load with the given type and
/// encoding. Higher values indicate more HW cost. Used for cost modeling in
/// RemoveLayoutConversions. Returns a comparable scalar (not cycle-accurate).
unsigned estimateLoadHWCost(RankedTensorType type, Operation *loadOp);

/// Return the canonical memory-coalesced layout that tritongpu-coalesce assigns
/// to a descriptor load/store operand of the given type — sizePerThread
/// vectorized along the contiguous (last) dim, row-major order, warps/threads
/// distributed by BlockedEncodingAttr::get. RemoveLayoutConversions uses this
/// to recognize (and preserve) a Coalesce-inserted convert feeding a
/// tt.descriptor_store: folding it back to the producer's compute layout
/// demotes the store's global coalescing (issue #7104).
Attribute canonicalCoalescedDescStoreLayout(RankedTensorType type, int numWarps,
                                            int threadsPerWarp);

/// Runtime stride value along `dim`; null if none.
Value getRuntimeStrideValue(triton::intel::ModuleStrideAnalysis &strideAnalysis,
                            Value ptr, unsigned dim);

/// Build `stride * elemSizeInBytes` as an i32; null for non-integer stride.
Value materializePitchBytes(OpBuilder &builder, Location loc, Value stride,
                            unsigned elemSizeInBytes);

/// True when `other` is absent or a splat of positive zero -- the only fill
/// value the 2D block message's hardware boundary check can produce. `-0.0`
/// is excluded even though `APFloat::isZero()` accepts it, since dropping
/// `other = -0.0` would flip masked-out elements' sign bit.
bool isZeroFillOther(Value other);

/// A mask recognized as a "row-limit" pattern: `(rowBase + arange(0, N)) <
/// bound`, constant across the tensor's other dimension -- e.g. a
/// grouped-GEMM row mask `offs_am < tl.load(group_end_ptr)`. Expressible as
/// a 2D block load's `base_height` operand instead of per-element
/// predication, since the hardware already zero-fills rows past it.
struct RowLimitMaskInfo {
  unsigned boundedDim; // tensor dim the mask varies (and is bounded) along
  Value rowBase;       // scalar base of the linear index; null means zero
  Value bound;         // scalar upper bound compared against
};

/// Match `mask` against the row-limit pattern above for a rank-2 tensor of
/// the given shape, and verify every other condition the `base_height`
/// conversion needs: `other` is zero-fill-compatible, the bounded dim is
/// the pitch dim (complement of `contiguousDim`, not the memory-contiguous
/// one), and the pitch dim isn't a stride-zero (broadcast) pointer, which
/// relies on a fixed physical height instead. Also requires `rowBase`/`bound`
/// to be exactly i32, as the `base_height` lowering needs.
///
/// All three callers (MaterializeBlockPointer's `maskPermitsBlockTile`,
/// LowerTo2DBlockLoad, and the prefetch LLVM lowering) must agree on
/// eligibility for the same op, or a load gets `ttig.block_io` stamped on a
/// promise a later pass refuses, de-anchoring it (see
/// `isExpensiveLoadOrStore`) without ever gaining a real 2D block read.
/// Centralizing every check here is what keeps them from drifting apart.
///
/// Returns std::nullopt otherwise, in which case the caller must still use
/// per-element masking.
std::optional<RowLimitMaskInfo> matchRowLimitMask(Value mask, Value other,
                                                  ArrayRef<int64_t> tensorShape,
                                                  unsigned contiguousDim,
                                                  bool pitchDimIsBroadcast);

/// Compute `clamp(bound - rowBase, 0, rowExtent)` as an i32. Widens to i64
/// first so the subtraction can't overflow (both are i32 and may be far
/// apart, e.g. `rowBase = -1`, `bound = INT_MAX`). `rowBase` may be null.
Value computeRowLimitHeight(OpBuilder &builder, Location loc, Value bound,
                            Value rowBase, int64_t rowExtent);

} // namespace mlir::triton::gpu::intel

#endif // TRITONINTELGPU_TRANSFORMS_BLOCKIOUTILS_H
