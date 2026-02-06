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
  // Select load operations that use a block ptr (defined by a MakeTensorPtrOp)
  // or a tensor descriptor (defined by a MakeTensorDescOp) with at least one
  // stride that has unknown value at compile time.
  virtual bool isCandidate(scf::ForOp &forOp, Operation *op) const {
    assert(op->getParentOfType<scf::ForOp>() == forOp &&
           "Operation should be contains in the loop");
    return TypeSwitch<Operation *, bool>(op)
        .Case<tt::LoadOp, tt::DescriptorLoadOp>(
            [this](auto loadOp) { return isCandidate(loadOp); })
        .Default([](auto) { return false; });
  }

  std::string getName() const { return "OpSelector"; };

private:
  bool isCandidate(tt::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr);
    if (!makeTensorPtrOp)
      return false;

    Operation::operand_range strides = makeTensorPtrOp->getStrides();
    if (strides.size() > 2)
      return false;

    return noStrideIsOne(strides);
  }

  bool isCandidate(tt::DescriptorLoadOp loadOp) const {
    std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(loadOp.getDesc());
    if (!makeTensorDescOp)
      return false;

    Operation::operand_range strides = makeTensorDescOp->getStrides();
    if (strides.size() > 2)
      return false;

    return noStrideIsOne(strides);
  }

  bool noStrideIsOne(OperandRange strides) const {
    auto isOne = [](Value v) {
      if (auto constantOp = v.getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr()))
          return intAttr.getInt() == 1;
      return false;
    };

    return llvm::none_of(strides, [&](Value stride) {
      Value finalVal = tt::intel::getFinalValue(stride);
      assert(finalVal && "Expecting a valid value");
      return finalVal.getDefiningOp() && isOne(finalVal);
    });
  }
};

// Collects operations in a loop that satisfy the condition imposed by
// the operation selector associated with this class.
template <typename OpSelector> class OpsCollector {
public:
  using Operations = SmallPtrSet<Operation *, 8>;

  OpsCollector(scf::ForOp &forOp, OpSelector &selector)
      : forOp(forOp), selector(selector) {}

  unsigned collectOps() {
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
    collectOps(forOp.getOps<tt::DescriptorLoadOp>(), ops);
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
  void version(scf::ForOp &forOp, ArrayRef<Operation *> ops,
               std::unordered_map<Operation *, Value> opToStride) const {
    assert(!ops.empty() && ops.size() == opToStride.size() &&
           "Sizes should match");

    Location loc = forOp.getLoc();
    auto funcOp = forOp->getParentOfType<tt::FuncOp>();
    OpBuilder builder(&funcOp.front().front());
    auto oneVal =
        arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1));

    // Build the versioning condition for the loop.
    builder.setInsertionPoint(forOp);
    SmallVector<Value> versioningConds;
    for (Operation *op : ops) {
      Value stride = opToStride[op];
      versioningConds.emplace_back(arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, stride, oneVal));
    }
    assert(!versioningConds.empty() &&
           "Expecting at least one versioning condition");

    Value verCond = versioningConds.front();
    for (unsigned i = 1; i < versioningConds.size(); ++i)
      verCond =
          arith::AndIOp::create(builder, loc, verCond, versioningConds[i]);

    // Version the loop.
    auto ifOp = scf::IfOp::create(builder, loc, forOp.getResultTypes(), verCond,
                                  /*withThenRegion=*/true);
    IRMapping map;
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    OpBuilder elseB = ifOp.getElseBodyBuilder();
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Create the yield operations for the two if branches.
    if (!thenForLoop->getResults().empty()) {
      scf::YieldOp::create(thenB, loc, thenForLoop->getResults());
      scf::YieldOp::create(elseB, loc, elseForLoop->getResults());
    }

    // Now that the loop has been versioned, replace the uses of the original
    // loop results.
    for (const auto &[i, v] : llvm::enumerate(forOp.getResults()))
      if (!v.getUsers().empty())
        v.replaceAllUsesWith(ifOp->getResult(i));

    auto updateStride = [&](MutableOperandRange strides, Value versionedStride,
                            Value oneVal) {
      for (OpOperand &stride : strides) {
        if (stride.get() == versionedStride) {
          stride.set(oneVal);
          break;
        }
      }
    };

    // Clone the original operation and replace the 'versioned' stride with one.
    for (Operation *op : ops) {
      Operation *newOp = op->clone();
      Value versionedStride = opToStride[op];
      if (isa<tt::MakeTensorPtrOp>(op))
        updateStride(cast<tt::MakeTensorPtrOp>(newOp).getStridesMutable(),
                     versionedStride, oneVal);
      else if (isa<tt::MakeTensorDescOp>(op))
        updateStride(cast<tt::MakeTensorDescOp>(newOp).getStridesMutable(),
                     versionedStride, oneVal);
      else
        llvm_unreachable("Unexpected operation type");

      builder.setInsertionPoint(op);
      builder.insert(newOp);
      op->replaceUsesWithIf(newOp, [&](OpOperand &use) {
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
        if (!isCandidateLoop(forOp))
          return WalkResult::advance();

        // Collect candidate operations. These are load/store operations that
        // use a block ptr with no stride equal to one (at compile time).
        OpSelector selector;
        OpsCollector collector(forOp, selector);
        if (collector.collectOps() == 0)
          return WalkResult::advance();

        OpBuilder builder(forOp);
        SmallVector<Operation *> selectedOps;
        std::unordered_map<Operation *, Value> selectedOpToStride;
        for (Operation *op : collector.getOps()) {
          TypeSwitch<Operation *>(op)
              .Case<tt::LoadOp>([&](auto loadOp) {
                processLoad(loadOp, selectedOps, selectedOpToStride);
              })
              .Case<tt::DescriptorLoadOp>([&](auto loadOp) {
                processDescLoad(loadOp, selectedOps, selectedOpToStride);
              })
              .Default([](auto) { return false; });
        }

        if (!selectedOpToStride.empty()) {
          LoopVersioner loopVersioner;
          loopVersioner.version(forOp, selectedOps, selectedOpToStride);
        }
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  bool isCandidateLoop(scf::ForOp forOp) const {
    // Nested loop aren't currently handled.
    if (forOp->getParentOfType<scf::ForOp>())
      return false;

    // Consider only loops with a single IV.
    if (!forOp.getSingleInductionVar())
      return false;
    return true;
  }

  void processLoad(
      tt::LoadOp loadOp, SmallVectorImpl<Operation *> &selectedOps,
      std::unordered_map<Operation *, Value> &selectedOpToStride) const {
    Value ptr = loadOp.getPtr();
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expecting a tensor pointer");

    auto makeTensorPtrOp =
        *tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr);
    OperandRange strides = makeTensorPtrOp.getStrides();
    ArrayRef<int> order = makeTensorPtrOp.getOrder();

    for (size_t idx = 0; idx < order.size(); ++idx) {
      unsigned strideIdx = order[idx];
      Value finalVal = tt::intel::getFinalValue(strides[strideIdx]);
      assert(finalVal && "Expecting a valid value");
      if (isFuncArgWithValueMaybeEqualToOne(finalVal)) {
        selectedOpToStride[makeTensorPtrOp] = cast<BlockArgument>(finalVal);
        break;
      }
    }

    if (selectedOpToStride.count(makeTensorPtrOp) != 0)
      selectedOps.push_back(makeTensorPtrOp);
  }

  void processDescLoad(
      tt::DescriptorLoadOp descLoadOp,
      SmallVectorImpl<Operation *> &selectedOps,
      std::unordered_map<Operation *, Value> &selectedOpToStride) const {
    auto makeTensorDescOp =
        *tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
            descLoadOp.getDesc());

    OperandRange strides = makeTensorDescOp.getStrides();
    for (Value stride : llvm::reverse(strides)) {
      Value finalVal = tt::intel::getFinalValue(stride);
      assert(finalVal && "Expecting a valid value");
      if (isFuncArgWithValueMaybeEqualToOne(finalVal)) {
        selectedOpToStride[makeTensorDescOp] = cast<BlockArgument>(finalVal);
        break;
      }
    }

    if (selectedOpToStride.count(makeTensorDescOp) != 0)
      selectedOps.push_back(makeTensorDescOp);
  }

  bool isFuncArgWithValueMaybeEqualToOne(Value stride) const {
    Value finalVal = tt::intel::getFinalValue(stride);
    assert(finalVal && "Expecting a valid value");

    Operation *defOp = finalVal.getDefiningOp();
    if (defOp)
      return false;

    auto blockArg = cast<BlockArgument>(finalVal);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    auto funcOp = dyn_cast<tt::FuncOp>(parentOp);
    if (!funcOp)
      return false;

    // Function arguments that have a divisibility attribute (e.g. by 16)
    // cannot have value equal to one (note: divisibility attributes cannot be
    // one).
    auto divisibilityAttr = funcOp.getArgAttrOfType<IntegerAttr>(
        blockArg.getArgNumber(), "tt.divisibility");
    if (divisibilityAttr) {
      assert(divisibilityAttr.getValue().isStrictlyPositive() &&
             !divisibilityAttr.getValue().isOne() &&
             "Unexpected divisibility value");
      return false;
    }

    return true;
  }
};

} // namespace
