#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-optimize-dot-operands"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEDOTOPERANDS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Transform:
//   %ptr = make_block_ptr [shapeN, shapeK], [strideN, strideK], [offN, offK]
//        : tt.ptr<tensor<NxK, enc>
//   %load = tt.load %ptr, {blockIO=<row_major|column_major>}
//         : tt.ptr<tensor<NxK, enc>
//   %trans = tt.trans %load : tt.ptr<tensor<KxN, dotEnc>>
//   tt.dot(%a, %trans)
// into:
//   %ptr = make_block_ptr [shapeK, shapeN], [strideK, strideN], [offK, offN]
//        : tt.ptr<tensor<KxN, dotEnc>
//   %load = tt.load %ptr, {blockIO=<column_major|row_major>}
//         : tt.ptr<tensor<KxN, dotEnc>
//   tt.dot(%a, %load)
class FuseTransWithLoad : public OpRewritePattern<tt::TransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::TransOp transOp,
                                PatternRewriter &rewriter) const override {
    if (!isCandidate(transOp))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Candidate: " << transOp << "\n");
    auto tensorType = cast<RankedTensorType>(transOp.getType());
    Attribute dotEncoding =
        cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
    auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
    tt::MakeTensorPtrOp makeTensorPtrOp =
        *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    LLVM_DEBUG(llvm::dbgs() << "makeTensorPtrOp: " << makeTensorPtrOp << "\n");

    // Create a MakeTensorPtrOp yielding a block pointer to the transposed
    // tensor.
    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto newPtrType =
        tt::PointerType::get(tensorType, ptrType.getAddressSpace());
    SmallVector<Value> newShape(llvm::reverse(makeTensorPtrOp.getShape()));
    SmallVector<Value> newStrides(llvm::reverse(makeTensorPtrOp.getStrides()));
    SmallVector<Value> newOffsets(llvm::reverse(makeTensorPtrOp.getOffsets()));

    OpBuilder builder(makeTensorPtrOp);
    Value ptr = builder.create<tt::MakeTensorPtrOp>(
        makeTensorPtrOp.getLoc(), newPtrType, makeTensorPtrOp.getBase(),
        newShape, newStrides, newOffsets, makeTensorPtrOp.getOrderAttr());
    assert(makeTensorPtrOp->hasOneUse() && "Expecting single user");
    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp: " << ptr << "\n");

    // Transitively update users of the block pointer.
    Operation *makeTensorPtrOpUser = *makeTensorPtrOp->getUsers().begin();
    if (auto advanceOp = dyn_cast<tt::AdvanceOp>(makeTensorPtrOpUser)) {
      ptr = updateAdvanceOpChain(advanceOp, loadOp, ptr);
    } else {
      // TODO: handle loop init args (scf.for only for now).
      assert(makeTensorPtrOpUser == loadOp &&
             "Expecting the load to be the user");
    }

    // Replace the load+transpose with a new load operation that uses the
    // transposed block pointer.
    auto newLoadOp = rewriter.create<tt::LoadOp>(
        loadOp.getLoc(), ptr, loadOp.getMask(), loadOp.getOther(),
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

    StringRef blockIOAttrName =
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
    StringAttr attr = loadOp->getAttrOfType<StringAttr>(blockIOAttrName);
    StringAttr newAttr =
        (attr == "row_major")
            ? StringAttr::get(loadOp->getContext(), "column_major")
        : (attr == "column_major")
            ? StringAttr::get(loadOp->getContext(), "row_major")
            : nullptr;
    assert(newAttr && "Expecting a valid blockIO attribute");

    newLoadOp->setAttr(blockIOAttrName, newAttr);
    LLVM_DEBUG(llvm::dbgs() << "newLoadOp: " << newLoadOp << "\n");

    transOp->replaceAllUsesWith(newLoadOp);

    return success();
  }

private:
  // Candidate is of the form:
  //   tt.dot(tt.trans(tt.load(..., {blockIO=...})))
  // Where:
  //  - the transpose result is used only by the dot operation, and
  //  - the transpose operation uses the result of a 2-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` in the same
  //    function, and
  //  - each operation in the def-use chain origination at the `make_tensor_ptr`
  //    and terminating at the load has a single user.
  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    bool transOpUsedOnlyByDotOp =
        transOp->hasOneUse() &&
        isa<triton::DotOp>(*transOp->getUsers().begin());
    Attribute transOpEncoding = transOp.getType().getEncoding();
    if (!transOpUsedOnlyByDotOp || !transOpEncoding ||
        !isa<ttg::DotOperandEncodingAttr>(transOpEncoding))
      return false;

    Operation *defOp = transOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    return isCandidate(cast<tt::LoadOp>(defOp));
  }

  bool isCandidate(tt::LoadOp loadOp) const {
    assert(loadOp && "Expecting a valid load operation");

    bool loadOpHasBlockIOAttr = loadOp->hasAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    if (!loadOp->hasOneUse() || !loadOpHasBlockIOAttr)
      return false;

    auto ptrType = cast<tt::PointerType>(loadOp.getPtr().getType());
    if (!isTensorPointerType(ptrType) ||
        cast<RankedTensorType>(ptrType.getPointeeType()).getRank() != 2)
      return false;

    std::optional<tt::MakeTensorPtrOp> defOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    if (!defOp || !singleUsersInChain(*defOp, loadOp))
      return false;

    return true;
  }

  bool singleUsersInChain(Operation *start, Operation *end) const {
    assert(start && end && "Expecting valid operations");
    Operation *currentOp = start;

    while (currentOp != end) {
      // TODO: extend to handle loops.
      if ((currentOp->getNumRegions() != 0) || !currentOp->hasOneUse())
        return false;

      currentOp = *currentOp->getUsers().begin();
    }

    return true;
  }

  // Recursively update the operands in a chain of AdvanceOps, after setting the
  // pointer operand of the first one.
  tt::AdvanceOp updateAdvanceOpChain(tt::AdvanceOp advanceOp, tt::LoadOp loadOp,
                                     Value ptr) const {
    assert(advanceOp->hasOneUse() && "Expecting single user");
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expecting a block pointer");

    Operation *user = *advanceOp->getUsers().begin();
    if (auto loadUser = dyn_cast<tt::LoadOp>(user)) {
      assert(loadUser == loadOp &&
             "chain should be terminated by candidate load");
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      return rewriter.create<tt::AdvanceOp>(advanceOp.getLoc(), ptr.getType(),
                                            ptr, newOffsets);
    }

    if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user)) {
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      ptr = rewriter.create<tt::AdvanceOp>(advanceOp.getLoc(), ptr.getType(),
                                           ptr, newOffsets);
      return updateAdvanceOpChain(advanceOp, loadOp, ptr);
    }

    llvm_unreachable("Unexpected user");
    return nullptr;
  }
};

} // namespace

class TritonIntelGPUOptimizeDotOperandsPass
    : public triton::gpu::intel::impl::TritonIntelGPUOptimizeDotOperandsBase<
          TritonIntelGPUOptimizeDotOperandsPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUOptimizeDotOperandsBase<
      TritonIntelGPUOptimizeDotOperandsPass>::
      TritonIntelGPUOptimizeDotOperandsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<FuseTransWithLoad>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};
