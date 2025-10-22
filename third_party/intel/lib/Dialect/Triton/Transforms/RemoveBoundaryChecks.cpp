#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/WalkResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <optional>

#define DEBUG_TYPE "triton-intel-remove-boundary-checks"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELREMOVEBOUNDARYCHECKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {
class BoundaryChecksRemover {
public:
  void run(ModuleOp moduleOp) {
    moduleOp.walk([&](tt::LoadOp loadOp) {
      if (!isCandidate(loadOp))
        return WalkResult::skip();

      tt::MakeTensorPtrOp makeTensorPtrOp =
          *tt::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
      LLVM_DEBUG(llvm::dbgs()
                 << "Analyzing boundaryCheck for: " << loadOp << "\n");

      SmallVector<int> newBoundaryCheck;
      for (int boundIdx : loadOp.getBoundaryCheck()) {
        ArrayRef<int> order = makeTensorPtrOp.getOrder();
        int idx = order.size() - order[boundIdx] - 1;
        Value offset = makeTensorPtrOp.getOffsets()[idx];
        Value shape = makeTensorPtrOp.getShape()[idx];
        std::optional<int64_t> offsetVal = getConstantIntValue(offset),
                               shapeVal = getConstantIntValue(shape);

        // If the shape is not known at compile time we cannot determine whether
        // the bound check is unnecessary.
        if (!shapeVal) {
          LLVM_DEBUG(llvm::dbgs().indent(2)
                     << "Check at index " << boundIdx << " is necessary\n");
          newBoundaryCheck.push_back(boundIdx);
          continue;
        }

        // Case 1: offset and shape are constant.
        if (offsetVal && *offsetVal < *shapeVal) {
          LLVM_DEBUG(llvm::dbgs().indent(2)
                     << "Check at index " << boundIdx << " is unnecessary\n");
          continue;
        }

        // Case 2: analyze boundary check in loops.
        if (auto forOp = makeTensorPtrOp->getParentOfType<scf::ForOp>()) {
          assert(forOp.getSingleInductionVar() && "Single IV expected");
          Value iv = *forOp.getSingleInductionVar();
          if (offset != iv) {
            LLVM_DEBUG(llvm::dbgs().indent(2)
                       << "Check at index " << boundIdx << " is necessary\n");
            newBoundaryCheck.push_back(boundIdx);
            continue;
          }

          OpFoldResult lb = *forOp.getSingleLowerBound();
          OpFoldResult ub = *forOp.getSingleUpperBound();
          OpFoldResult step = *forOp.getSingleStep();

          auto computeLoopIVRange =
              [&](OpFoldResult lb, OpFoldResult ub,
                  OpFoldResult step) -> std::optional<ConstantIntRanges> {
            auto getBoundValue =
                [](OpFoldResult bound) -> std::optional<int64_t> {
              if (std::optional<int64_t> opVal = getConstantIntValue(bound))
                return *opVal;

              Value val = tt::intel::getFinalValue(cast<Value>(bound));
              if (auto cst = dyn_cast<arith::BitcastOp>(val.getDefiningOp()))
                val = cst.getIn();

              return getConstantIntValue(getAsOpFoldResult(val));
            };

            auto areLoopBoundKnown = [&](OpFoldResult lb, OpFoldResult ub,
                                         OpFoldResult step) {
              return (getBoundValue(lb) && getBoundValue(ub) &&
                      getBoundValue(step));
            };

            if (!areLoopBoundKnown(lb, ub, step))
              return std::nullopt;

            int64_t lbVal = *getBoundValue(lb);
            int64_t ubVal = *getBoundValue(ub);
            int64_t stepVal = *getBoundValue(step);
            int64_t lastIVVal =
                lbVal + ((ubVal - lbVal - 1) / stepVal) * stepVal;
            llvm::APInt start(64, lbVal, true);
            llvm::APInt end(64, lastIVVal, true);

            return ConstantIntRanges::range(start, end, true);
          };

          std::optional<ConstantIntRanges> optRange =
              computeLoopIVRange(lb, ub, step);
          if (!optRange) {
            LLVM_DEBUG(llvm::dbgs().indent(2)
                       << "Check at index " << boundIdx << " is necessary\n");
            newBoundaryCheck.push_back(boundIdx);
            continue;
          }

          // Compare the max value of the loop IV to the offset.
          APInt max = (*optRange).smax();
          if (max.getSExtValue() < shapeVal) {
            LLVM_DEBUG(llvm::dbgs().indent(2)
                       << "Check at index " << boundIdx << " is unnecessary\n");
            continue;
          }
        }

        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Check at index " << boundIdx << " is necessary\n");
        newBoundaryCheck.push_back(boundIdx);
      }

      if (newBoundaryCheck.size() != loadOp.getBoundaryCheck().size()) {
        loadOp.setBoundaryCheck(newBoundaryCheck);
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Rewritten load is: " << loadOp << "\n");
      }

      return WalkResult::advance();
    });
  }

private:
  // A candidate load operation is one that:
  //  - has the boundary check attribute
  //  - uses a block pointer defined by a `make_tensor_ptr` that is not
  //  advanced
  bool isCandidate(tt::LoadOp loadOp) const {
    assert(loadOp && "Expecting a valid load operation");

    ArrayRef<int> boundaryCheck = loadOp.getBoundaryCheck();
    if (boundaryCheck.empty())
      return false;

    Type ptrType = loadOp.getPtr().getType();
    if (!tt::isTensorPointerType(ptrType))
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        tt::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    if (!makeTensorPtrOp)
      return false;

    if (llvm::any_of((*makeTensorPtrOp)->getUsers(),
                     [](Operation *user) { return isa<tt::AdvanceOp>(user); }))
      return false;

    return true;
  }
};

} // namespace

struct TritonIntelRemoveBoundaryChecks
    : tt::intel::impl::TritonIntelRemoveBoundaryChecksBase<
          TritonIntelRemoveBoundaryChecks> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    BoundaryChecksRemover remover;
    remover.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};
