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
class ModuleAxisInfoAnalysis;
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

/// Return a blocked layout under which a row-major rank-2 load of \p type
/// passes validate2DBlockLoadTile, or std::nullopt. Lanes span at most 64
/// bytes of a row, and warps split the dim other than \p reducedDim (the
/// columns if none) first, to keep a reduction in a warp. \p type must have an
/// encoding, whose CGA layout is kept.
std::optional<triton::gpu::BlockedEncodingAttr>
pickBlockIOLoadEncoding(RankedTensorType type, unsigned numWarps,
                        unsigned threadsPerWarp,
                        std::optional<unsigned> reducedDim);

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

/// Return true if LowerTo2DBlockLoad would rewrite \p op to ttig.2d_block_load
/// with result type \p tensorTy, which may differ from op.getType(). Emits no
/// diagnostics and skips the 16-byte pitch check, which the pass reports as an
/// error, so true means "eligible", not "safe to emit" for hand-tagged IR.
bool isDescriptorLoadLowerableTo2DBlock(
    triton::DescriptorLoadOp op, RankedTensorType tensorTy,
    triton::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis);

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

/// Whether a 2D block surface supports this row pitch. Shared by
/// MaterializeBlockPointer and LowerTo2DBlockLoad so they cannot drift.
/// A constant pitch must be >= 64 bytes, a multiple of 16 and fit in 24 bits.
/// A runtime pitch needs a stride value and a full row of >= 64 bytes.
bool isBlockPitchSupported(std::optional<int64_t> constPitchBytes,
                           bool hasRuntimeStride, int64_t fullRowBytes);

/// Build `stride * elemSizeInBytes` as an i32; null for non-integer stride.
Value materializePitchBytes(OpBuilder &builder, Location loc, Value stride,
                            unsigned elemSizeInBytes);

/// True when `other` is absent or a positive-zero splat, the only fill value
/// the 2D block message's HW boundary check can produce. `-0.0` is excluded
/// to avoid flipping masked-out elements' sign bit.
bool isZeroFillOther(Value other);

/// A mask of the form `(rowBase + arange(0, N)) < bound`, broadcast along
/// every other dim (contiguous dim and any batch dims). Expressible as a
/// 2D block load's `base_height` instead of per-element predication.
///
/// `extraPred`, if non-null, is a batch-dims-only predicate ANDed into the
/// original mask (degenerate along the pitch and contiguous dims, so it's
/// uniform per hardware block-load message); the caller ANDs it into its
/// own per-message select instead of folding it into `base_height`. When the
/// AND sits below the final broadcast (`tt-reorder-broadcast`), `extraPred`
/// has that narrower shape rather than the mask's; use
/// `materializeExtraPred` to get a full-shape value.
struct RowLimitMaskInfo {
  unsigned boundedDim;
  Value rowBase;
  Value bound;
  Value extraPred;
};

/// Match `mask` against the row-limit pattern above for a tensor of any
/// rank >= 2: `matchBoundaryCheckMask` with at most one uniform conjunct, the
/// one `extraPred` holds.
std::optional<RowLimitMaskInfo> matchRowLimitMask(Value mask, Value other,
                                                  ArrayRef<int64_t> tensorShape,
                                                  unsigned contiguousDim,
                                                  bool pitchDimIsBroadcast);

/// A row-limit mask (see `RowLimitMaskInfo`) matched conjunct by conjunct:
/// an AND of a pitch-dim bound `(base + arange(0, extent)) < bound` and any
/// number of predicates uniform along both surface dims. The bound maps onto
/// the 2D block message's own `base_height` clamp, `clamp(bound - base, 0,
/// extent)` (see `computeRowLimitHeight`), so the mask needs no per-element
/// predication at all. `rowBase` is null when the index is a bare `arange`.
///
/// `extraPreds` are the uniform conjuncts (batch-dim predicates, or scalar
/// flags splatted to the mask's shape), each as `extraPred` in
/// `RowLimitMaskInfo`: the caller ANDs them into its own per-message select.
struct BoundaryMaskInfo {
  unsigned boundedDim; // the pitch dim, as in `RowLimitMaskInfo`
  Value rowBase, rowBound;
  SmallVector<Value> extraPreds;
};

/// Match `mask` against the pattern above for a tensor of any rank >= 2.
/// Fails unless every conjunct is recognised, `other` is
/// zero-fill-compatible, the pitch dim is not broadcast, and the bound is i32
/// or i64 with the base of the compare's own type (no extension between them).
/// The bound may have the index on either side (`bound > index`), and may be
/// ANDed with `index >= 0`, which is dropped only where it is vacuous: the
/// surface starts at the tensor's first row, so a row before it is read like
/// any other. \p loadOp is where the index base's sign is asked (see
/// `triton::valueIsNonNegative`); without it only a bare `arange` qualifies.
std::optional<BoundaryMaskInfo>
matchBoundaryCheckMask(Value mask, Value other, ArrayRef<int64_t> tensorShape,
                       unsigned contiguousDim, bool pitchDimIsBroadcast,
                       Operation *loadOp);

/// Return `extraPred` as a value of type \p maskTy: unchanged if it already has
/// that shape, else broadcast to it straight from its innermost source (so
/// no broadcast-of-broadcast is left behind).
Value materializeExtraPred(OpBuilder &builder, Location loc, Value extraPred,
                           RankedTensorType maskTy);

/// Compute `clamp(bound - rowBase, 0, rowExtent)` as an i32. `bound` and
/// `rowBase` are i32 or i64; the result is exact for any value, including i64
/// inputs whose difference overflows. `rowBase` may be null.
Value computeRowLimitHeight(OpBuilder &builder, Location loc, Value bound,
                            Value rowBase, int64_t rowExtent);

} // namespace mlir::triton::gpu::intel

#endif // TRITONINTELGPU_TRANSFORMS_BLOCKIOUTILS_H
