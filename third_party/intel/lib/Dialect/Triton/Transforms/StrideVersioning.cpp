#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

#define DEBUG_TYPE "triton-intel-stride-versioning"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELSTRIDEVERSIONING
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

class OpSelector {
public:
  // Select load operations that use a block ptr defined by a
  // make_tensor_ptr operation with at least one stride that has unknown value
  // at compile time.
  virtual bool isCandidate(scf::ForOp &forOp, Operation *op) const {
    assert(op->getParentOfType<scf::ForOp>() == forOp &&
           "Operation should be contains in the loop");

    return TypeSwitch<Operation *, bool>(op)
        .Case<tt::LoadOp>([](auto loadOp) {
          Value ptr = loadOp.getPtr();
          if (!tt::isTensorPointerType(ptr.getType()))
            return false;

          auto tensorType = cast<RankedTensorType>(
              cast<tt::PointerType>(ptr.getType()).getPointeeType());
          if (tensorType.getRank() > 2)
            return false;

          auto isOne = [](Value v) {
            auto constantOp = v.getDefiningOp<arith::ConstantOp>();
            if (!constantOp)
              return false;
            if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr()))
              return intAttr.getInt() == 1;
            return false;
          };

          // If no stride has value equal to one we have found a candidate
          // operation.
          tt::MakeTensorPtrOp makeTensorPtrOp = tt::getMakeTensorPtrOp(ptr);
          bool isCandidate =
              llvm::none_of(makeTensorPtrOp.getStrides(), [&](Value stride) {
                Value finalVal = tt::intel::getFinalValue(stride);
                assert(finalVal && "Expecting a valid value");
                return finalVal.getDefiningOp() && isOne(finalVal);
              });
          return isCandidate;
        })
        .Default([](auto) { return false; });
  }

  std::string getName() const { return "OpSelector"; };
};

// Collects operations in a loop that satisfy the condition imposed by
// the operation selector associated with this class.
template <typename OpSelector> class OpsCollector {
public:
  using Operations = SmallPtrSet<Operation *, 8>;

  OpsCollector(scf::ForOp &forOp, OpSelector &selector)
      : forOp(forOp), selector(selector) {}

  bool collectOps() {
    auto collectOps = [&](auto ops, Operations &selectedOps) {
      for (Operation *op : ops) {
        if (selector.isCandidate(forOp, op)) {
          selectedOps.insert(op);
          LLVM_DEBUG(llvm::dbgs()
                     << selector.getName() << ": collected: " << *op << "\n");
        }
      }
    };

    collectOps(forOp.getOps<tt::LoadOp>(), ops);
    return ops.size();
  }

  const Operations &getOps() const { return ops; };
  const OpSelector &getSelector() const { return selector; }

private:
  scf::ForOp &forOp;
  OpSelector &selector;
  Operations ops;
};

class LoopVersioner {
public:
  void
  version(scf::ForOp &forOp, ArrayRef<Operation *> makeTensorPtrOps,
          std::unordered_map<Operation *, Value> makeTensorPtrToStride) const {
    assert(!makeTensorPtrOps.empty() &&
           makeTensorPtrOps.size() == makeTensorPtrToStride.size() &&
           "Sizes should match");

    Location loc = forOp.getLoc();
    auto funcOp = forOp->getParentOfType<tt::FuncOp>();
    OpBuilder builder(&funcOp.front().front());
    auto oneVal =
        arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1));

    // Build the versioning condition for the loop.
    builder.setInsertionPoint(forOp);
    SmallVector<Value> versioningConds;
    for (Operation *makeTensorPtrOp : makeTensorPtrOps) {
      Value stride = makeTensorPtrToStride[makeTensorPtrOp];
      versioningConds.emplace_back(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, stride, oneVal));
    }
    assert(!versioningConds.empty() &&
           "Expecting at least one versioning condition");

    Value verCond = versioningConds.front();
    for (unsigned i = 1; i < versioningConds.size(); ++i)
      verCond = builder.create<arith::AndIOp>(loc, verCond, versioningConds[i]);

    // Version the loop.
    auto ifOp = builder.create<scf::IfOp>(loc, forOp.getResultTypes(), verCond,
                                          /*withThenRegion=*/true);
    IRMapping map;
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    OpBuilder elseB = ifOp.getElseBodyBuilder();
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Create the yield operations for the two if branches.
    if (!thenForLoop->getResults().empty()) {
      thenB.create<scf::YieldOp>(loc, thenForLoop->getResults());
      elseB.create<scf::YieldOp>(loc, elseForLoop->getResults());
    }

    // Now that the loop has been versioned, replace the uses of the original
    // loop results.
    for (const auto &[i, v] : llvm::enumerate(forOp.getResults()))
      if (!v.getUsers().empty())
        v.replaceAllUsesWith(ifOp->getResult(i));

    // Clone the makeTensorPtrOps and replace 'versioned' stride with one.
    for (Operation *makeTensorPtrOp : makeTensorPtrOps) {
      auto newOp = cast<tt::MakeTensorPtrOp>(makeTensorPtrOp->clone());
      Value versionedStride = makeTensorPtrToStride[makeTensorPtrOp];
      for (OpOperand &stride : newOp.getStridesMutable()) {
        if (stride.get() == versionedStride) {
          stride.set(oneVal);
          break;
        }
      }

      builder.setInsertionPoint(makeTensorPtrOp);
      builder.insert(newOp);
      makeTensorPtrOp->replaceUsesWithIf(newOp, [&](OpOperand &use) {
        return use.getOwner()->getParentOfType<scf::ForOp>() == thenForLoop;
      });
    }

    forOp.erase();
  }
};

struct TritonIntelStrideVersioning
    : tt::intel::impl::TritonIntelStrideVersioningBase<
          TritonIntelStrideVersioning> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        // Consider only loops with a single IV.
        if (!forOp.getSingleInductionVar())
          return WalkResult::advance();

        // Collect candidate operations. These are load/store operations that
        // use a block ptr with no stride equal to one (at compile time).
        OpSelector selector;
        OpsCollector collector(forOp, selector);
        if (collector.collectOps()) {
          OpBuilder builder(forOp);
          Location loc = forOp->getLoc();

          SmallVector<Operation *> makeTensorPtrOps;
          std::unordered_map<Operation *, Value> makeTensorPtrToStride;
          for (Operation *op : collector.getOps()) {
            TypeSwitch<Operation *>(op)
                .Case<tt::LoadOp>([&](auto loadOp) {
                  Value ptr = loadOp.getPtr();
                  assert(tt::isTensorPointerType(ptr.getType()) &&
                         "Expecting a block ptr");

                  tt::MakeTensorPtrOp makeTensorPtrOp =
                      tt::getMakeTensorPtrOp(ptr);
                  OperandRange strides = makeTensorPtrOp.getStrides();
                  ArrayRef<int> order = makeTensorPtrOp.getOrder();

                  for (size_t idx = 0; idx < order.size(); ++idx) {
                    unsigned strideIdx = order[idx];
                    Value stride = strides[strideIdx];
                    Value finalVal = tt::intel::getFinalValue(stride);
                    assert(finalVal && "Expecting a valid value");

                    Operation *defOp = finalVal.getDefiningOp();
                    if (defOp)
                      continue;

                    auto blockArg = cast<BlockArgument>(finalVal);
                    Operation *parentOp = blockArg.getOwner()->getParentOp();
                    auto funcOp = dyn_cast<tt::FuncOp>(parentOp);
                    if (!funcOp)
                      continue;

                    // arguments that have a divisibility attribute (e.g. by 16)
                    // cannot have value equal to one (the divisibility
                    // attribute should not be one).
                    auto divisibilityAttr =
                        funcOp.getArgAttrOfType<IntegerAttr>(
                            blockArg.getArgNumber(), "tt.divisibility");
                    if (divisibilityAttr) {
                      assert(divisibilityAttr.getValue().isStrictlyPositive() &&
                             !divisibilityAttr.getValue().isOne() &&
                             "Unexpected divisibility value");
                      continue;
                    }

                    makeTensorPtrToStride[makeTensorPtrOp] = blockArg;
                    break;
                  }

                  if (makeTensorPtrToStride.count(makeTensorPtrOp) != 0)
                    makeTensorPtrOps.push_back(makeTensorPtrOp);
                })
                .Default([](auto) { return false; });
          }

          if (!makeTensorPtrToStride.empty()) {
            LoopVersioner loopVersioner;
            loopVersioner.version(forOp, makeTensorPtrOps,
                                  makeTensorPtrToStride);
          }
        }
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
