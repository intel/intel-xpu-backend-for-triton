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

  StringAttr kWarp = StringAttr::get(ctx, "warp");
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

} // namespace mlir::triton::gpu::intel
