#include <triton/Dialect/Triton/IR/Utility.h>

#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "tritonintelgpu-sub32-type-optimization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
using LinearLayout = mlir::triton::LinearLayout;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUEMPTYANALYSIS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

/// Check whether a load is eligible for 2D block IO lowering.
template <typename OpTy,
          std::enable_if_t<
              llvm::is_one_of<OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value,
              bool> = true>
static bool isBlockIOEligible(OpTy op) {
  RankedTensorType tensorTy;
  if constexpr (std::is_same_v<OpTy, tt::LoadOp>) {
    tensorTy =
        dyn_cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
    if (!tensorTy)
      return false;
  } else {
    tensorTy = cast<RankedTensorType>(op.getType());
  }

  return ttgi::isBlockIOEligible(op, tensorTy);
}

class TritonIntelGPUEmptyAnalysisPass
    : public mlir::triton::gpu::intel::impl::TritonIntelGPUEmptyAnalysisBase<
          TritonIntelGPUEmptyAnalysisPass> {
public:
  using mlir::triton::gpu::intel::impl::TritonIntelGPUEmptyAnalysisBase<
      TritonIntelGPUEmptyAnalysisPass>::TritonIntelGPUEmptyAnalysisBase;

  void runOnOperation() override {

    ModuleOp mod = getOperation();

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    tt::intel::ModuleStrideAnalysis strideAnalysis(mod, axisInfoAnalysis);

    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&](tt::LoadOp op) { loadOps.push_back(op); });
    for (auto op : loadOps)
      convertLoadOp(op, strideAnalysis, axisInfoAnalysis);
  }

  static bool hasSupport256bLoadStore(Operation *op) {
    auto mod = op->getParentOfType<ModuleOp>();
    return mod &&
           mod->hasAttr(
               ttgi::TritonIntelGPUDialect::getSupport256bLoadStoreAttrName());
  }

  unsigned getContiguity(
      Value ptr,
      const tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisInfoAnalysis)
        .getContiguity(ptr);
  }

  unsigned getVectorSize(
      bool support256bLoadStore, Value ptr,
      const tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    if (!isa<RankedTensorType>(ptr.getType()))
      return 1;

    unsigned contiguity = getContiguity(ptr, axisInfoAnalysis);
    unsigned pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
    return std::min<unsigned>(
        getMaxVecWidth(support256bLoadStore, pointeeBitWidth), contiguity);
  }

  /// Maximum number of elements per-thread vector load/store. 128 bits by
  /// default; 256 bits when the target supports wider load/stores.
  static unsigned getMaxVecWidth(bool support256bLoadStore,
                                 unsigned pointeeBitWidth) {
    unsigned maxBits = support256bLoadStore ? 256u : 128u;
    return std::max(1u, maxBits / std::max(8u, pointeeBitWidth));
  }

  void
  convertLoadOp(tt::LoadOp op,
                const tt::intel::ModuleStrideAnalysis &strideAnalysis,
                const tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    if (isBlockIOEligible(op))
      return;

    MLIRContext *ctx = op->getContext();
    Value ptr = op.getPtr();
    unsigned vec =
        getVectorSize(hasSupport256bLoadStore(op), ptr, axisInfoAnalysis);
    // llvm::outs() << "johnlu op:" << op << "\n";
    Type valueElemTy = getElementTypeOrSelf(op.getType());
    auto tensorType = cast<RankedTensorType>(op.getType());
    Attribute encoding = tensorType.getEncoding();
    std::optional<LinearLayout> llEncoding =
        cast<ttg::DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    unsigned numElems = ttg::getTotalElemsPerThread(op.getType());
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;
    const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
    const size_t totalWidth = valueElemNBits * vec;
    const size_t width = std::min(totalWidth, maxWordWidth);
    const size_t nWords = std::max<size_t>(1, totalWidth / width);
    const size_t wordNElems = width / valueElemNBits;
    assert(wordNElems * nWords * numVecs == numElems);
    size_t packedElemsPerLane = mlir::ceil<size_t>(width, valueElemNBits);
    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    StringAttr kRegister = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");
    StringAttr kDim0 = StringAttr::get(ctx, "dim0");
    StringAttr kDim1 = StringAttr::get(ctx, "dim1");

    LinearLayout loadLayout =
        LinearLayout::identity1D(packedElemsPerLane, kRegister, kDim1) *
        LinearLayout::identity1D(threadsPerWarp / packedElemsPerLane, kLane,
                                 kDim1) *
        LinearLayout::identity1D(packedElemsPerLane, kLane, kDim0) *
        LinearLayout::identity1D(vec / packedElemsPerLane, kRegister, kDim0) *
        LinearLayout::identity1D(numElems / vec, kRegister, kDim0);
    // The SoA mapping of the non-uniform value of vector.
    LinearLayout reinterpretLayout =
        LinearLayout::identity1D(numElems, kRegister, kDim0) *
        LinearLayout::identity1D(threadsPerWarp, kLane, kDim1);
    // The cvtLayoutDstToSrcMap is: loadLayout =
    // cvtLayoutDstToSrcMap.compose(reinterpretLayout)
    LinearLayout cvtLayoutDstToSrcMap =
        reinterpretLayout.invertAndCompose(loadLayout);
    cvtLayoutDstToSrcMap *=
        LinearLayout::identity1D(llEncoding->getInDimSize(kWarp), kWarp, kWarp);
    cvtLayoutDstToSrcMap *= LinearLayout::identity1D(
        llEncoding->getInDimSize(kBlock), kBlock, kBlock);
    // The new layout is the cvtMapping to map the original layout with the new
    // register and lane mapping.
    LinearLayout newAoSLayout = cvtLayoutDstToSrcMap.compose(*llEncoding);

    auto reinterpretComp = llEncoding->invertAndCompose(newAoSLayout);

    OpBuilder builder(op);
    Location loc = op.getLoc();
    builder.setInsertionPointAfter(op);

    ttg::LinearEncodingAttr newLayout =
        ttg::LinearEncodingAttr::get(ctx, newAoSLayout);
    auto reinterpretedResult = ttgi::ReinterpretConvertLayoutOp::create(
        builder, loc, tensorType.cloneWithEncoding(newLayout), op.getResult());
    auto convertLayoutPair = ttg::ConvertLayoutOp::create(
        builder, loc, tensorType, reinterpretedResult.getResult());

    // Replace all uses except the one in reinterpretedResult to avoid
    // self-cycle.
    op.getResult().replaceUsesWithIf(
        convertLayoutPair.getResult(), [&](OpOperand &use) {
          return use.getOwner() != reinterpretedResult.getOperation();
        });
  }
};

} // namespace
