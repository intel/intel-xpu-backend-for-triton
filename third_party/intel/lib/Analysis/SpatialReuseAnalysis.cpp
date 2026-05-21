#include "intel/include/Analysis/SpatialReuseAnalysis.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gpu::intel {

static SmallVector<unsigned> fullAxisSet(unsigned rank) {
  SmallVector<unsigned> result;
  result.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    result.push_back(i);
  return result;
}

/// Build the LinearLayout for `ty` and verify it has a "warp" in-dim.
/// Returns std::nullopt on the same conditions that `knownWarpInvariantOutDims`
/// and `knownWarpBroadcastFactor` use to report "unknown":
///   - encoding is null,
///   - any shape dim is non-power-of-2 (toLinearLayout asserts),
///   - the layout has no "warp" in-dim.
/// Centralizes the prologue shared by the two `known*` accessors and
/// `getWarpInvariantOutDims` (the latter falls back to a full axis set on
/// std::nullopt; the former two propagate it).
static std::optional<LinearLayout> tryBuildWarpLayout(RankedTensorType ty,
                                                      StringAttr &kWarpOut) {
  Attribute enc = ty.getEncoding();
  if (!enc)
    return std::nullopt;

  for (int64_t dim : ty.getShape()) {
    if (!llvm::isPowerOf2_64(dim))
      return std::nullopt;
  }

  LinearLayout ll = ttg::toLinearLayout(ty);

  StringAttr kWarp = StringAttr::get(ty.getContext(), "warp");
  if (!ll.hasInDim(kWarp))
    return std::nullopt;

  kWarpOut = kWarp;
  return ll;
}

SmallVector<unsigned>
SpatialReuseAnalysis::getWarpInvariantOutDims(RankedTensorType ty) const {
  unsigned rank = ty.getRank();

  // No manual SliceEncodingAttr unwrap: SliceEncodingAttr::toLinearLayout
  // already handles removing the sliced dim internally and emits a
  // LinearLayout at the correct (sliced) rank. Manually unwrapping here
  // would apply the parent encoding at the wrong rank.

  StringAttr kWarp;
  std::optional<LinearLayout> ll = tryBuildWarpLayout(ty, kWarp);
  if (!ll)
    return fullAxisSet(rank);

  SmallVector<StringAttr> outDims = llvm::to_vector(ll->getOutDimNames());
  SmallVector<unsigned> result;
  for (unsigned i = 0; i < outDims.size(); ++i) {
    if (ll->sublayoutIsZero({kWarp}, {outDims[i]}))
      result.push_back(i);
  }
  return result;
}

bool SpatialReuseAnalysis::hasCrossSubgroupReuse(RankedTensorType ty) const {
  return !getWarpInvariantOutDims(ty).empty();
}

bool SpatialReuseAnalysis::hasCrossSubgroupReuse(tt::LoadOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  return rt ? hasCrossSubgroupReuse(rt) : true;
}

bool SpatialReuseAnalysis::hasCrossSubgroupReuse(
    tt::DescriptorLoadOp op) const {
  return hasCrossSubgroupReuse(cast<RankedTensorType>(op.getType()));
}

bool SpatialReuseAnalysis::hasCrossSubgroupReuse(
    tt::DescriptorGatherOp op) const {
  return hasCrossSubgroupReuse(cast<RankedTensorType>(op.getType()));
}

SmallVector<unsigned>
SpatialReuseAnalysis::getWarpInvariantOutDims(tt::LoadOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  return rt ? getWarpInvariantOutDims(rt) : SmallVector<unsigned>{};
}

SmallVector<unsigned>
SpatialReuseAnalysis::getWarpInvariantOutDims(tt::DescriptorLoadOp op) const {
  return getWarpInvariantOutDims(cast<RankedTensorType>(op.getType()));
}

SmallVector<unsigned>
SpatialReuseAnalysis::getWarpInvariantOutDims(tt::DescriptorGatherOp op) const {
  return getWarpInvariantOutDims(cast<RankedTensorType>(op.getType()));
}

std::optional<SmallVector<unsigned>>
SpatialReuseAnalysis::knownWarpInvariantOutDims(RankedTensorType ty) const {
  StringAttr kWarp;
  std::optional<LinearLayout> ll = tryBuildWarpLayout(ty, kWarp);
  if (!ll)
    return std::nullopt;

  SmallVector<StringAttr> outDims = llvm::to_vector(ll->getOutDimNames());
  SmallVector<unsigned> result;
  for (unsigned i = 0; i < outDims.size(); ++i) {
    if (ll->sublayoutIsZero({kWarp}, {outDims[i]}))
      result.push_back(i);
  }
  return result; // known, possibly empty.
}

std::optional<SmallVector<unsigned>>
SpatialReuseAnalysis::knownWarpInvariantOutDims(tt::LoadOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  if (!rt)
    return std::nullopt;
  return knownWarpInvariantOutDims(rt);
}

std::optional<unsigned>
SpatialReuseAnalysis::knownWarpBroadcastFactor(RankedTensorType ty) const {
  StringAttr kWarp;
  std::optional<LinearLayout> ll = tryBuildWarpLayout(ty, kWarp);
  if (!ll)
    return std::nullopt;

  // Each warp-bit whose entire basis vector is all-zero across out-dims is a
  // pure broadcast bit (incrementing it does not move the access). Count
  // those bits; the broadcast factor is 2^count.
  unsigned numBases = ll->getInDimSizeLog2(kWarp);
  unsigned zeroBases = 0;
  for (unsigned pos = 0; pos < numBases; ++pos) {
    ArrayRef<int32_t> basis = ll->getBasis(kWarp, pos);
    if (llvm::all_of(basis, [](int32_t v) { return v == 0; }))
      ++zeroBases;
  }
  return 1u << zeroBases;
}

std::optional<unsigned>
SpatialReuseAnalysis::knownWarpBroadcastFactor(tt::LoadOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  if (!rt)
    return std::nullopt;
  return knownWarpBroadcastFactor(rt);
}

} // namespace mlir::triton::gpu::intel
