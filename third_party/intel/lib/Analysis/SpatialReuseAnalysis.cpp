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

SmallVector<unsigned>
SpatialReuseAnalysis::getWarpInvariantOutDims(RankedTensorType ty) const {
  unsigned rank = ty.getRank();

  Attribute enc = ty.getEncoding();
  if (!enc)
    return fullAxisSet(rank);

  // No manual SliceEncodingAttr unwrap: SliceEncodingAttr::toLinearLayout
  // already handles removing the sliced dim internally and emits a
  // LinearLayout at the correct (sliced) rank. Manually unwrapping here
  // would apply the parent encoding at the wrong rank.

  // toLinearLayout asserts on non-power-of-2 shapes; bail conservatively.
  for (int64_t dim : ty.getShape()) {
    if (!llvm::isPowerOf2_64(dim))
      return fullAxisSet(rank);
  }

  LinearLayout ll = ttg::toLinearLayout(ty);

  StringAttr kWarp = StringAttr::get(ty.getContext(), "warp");
  if (!ll.hasInDim(kWarp))
    return fullAxisSet(rank);

  SmallVector<StringAttr> outDims = llvm::to_vector(ll.getOutDimNames());
  SmallVector<unsigned> result;
  for (unsigned i = 0; i < outDims.size(); ++i) {
    if (ll.sublayoutIsZero({kWarp}, {outDims[i]}))
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
  Attribute enc = ty.getEncoding();
  if (!enc)
    return std::nullopt;

  // toLinearLayout asserts on non-power-of-2 shapes; report unknown.
  for (int64_t dim : ty.getShape()) {
    if (!llvm::isPowerOf2_64(dim))
      return std::nullopt;
  }

  LinearLayout ll = ttg::toLinearLayout(ty);

  StringAttr kWarp = StringAttr::get(ty.getContext(), "warp");
  if (!ll.hasInDim(kWarp))
    return std::nullopt;

  SmallVector<StringAttr> outDims = llvm::to_vector(ll.getOutDimNames());
  SmallVector<unsigned> result;
  for (unsigned i = 0; i < outDims.size(); ++i) {
    if (ll.sublayoutIsZero({kWarp}, {outDims[i]}))
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

std::optional<SmallVector<unsigned>>
SpatialReuseAnalysis::knownWarpInvariantOutDims(tt::DescriptorLoadOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  if (!rt)
    return std::nullopt;
  return knownWarpInvariantOutDims(rt);
}

std::optional<SmallVector<unsigned>>
SpatialReuseAnalysis::knownWarpInvariantOutDims(
    tt::DescriptorGatherOp op) const {
  auto rt = dyn_cast<RankedTensorType>(op.getType());
  if (!rt)
    return std::nullopt;
  return knownWarpInvariantOutDims(rt);
}

bool SpatialReuseAnalysis::knownCrossSubgroupReuse(RankedTensorType ty) const {
  std::optional<SmallVector<unsigned>> dims = knownWarpInvariantOutDims(ty);
  return dims.has_value() && !dims->empty();
}

bool SpatialReuseAnalysis::knownCrossSubgroupReuse(tt::LoadOp op) const {
  std::optional<SmallVector<unsigned>> dims = knownWarpInvariantOutDims(op);
  return dims.has_value() && !dims->empty();
}

bool SpatialReuseAnalysis::knownCrossSubgroupReuse(
    tt::DescriptorLoadOp op) const {
  std::optional<SmallVector<unsigned>> dims = knownWarpInvariantOutDims(op);
  return dims.has_value() && !dims->empty();
}

bool SpatialReuseAnalysis::knownCrossSubgroupReuse(
    tt::DescriptorGatherOp op) const {
  std::optional<SmallVector<unsigned>> dims = knownWarpInvariantOutDims(op);
  return dims.has_value() && !dims->empty();
}

} // namespace mlir::triton::gpu::intel
