#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

using namespace mlir;
namespace tt = mlir::triton;

static std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
  return std::nullopt;
}

namespace mlir::triton::intel {

Value findOrCreateIntConstant(Location loc, int val, unsigned bitWidth,
                              OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto cstOp = dyn_cast<arith::ConstantIntOp>(op))
      return cstOp.value() == val &&
             cstOp.getType().getIntOrFloatBitWidth() == bitWidth;
    return false;
  });

  return (it != insertPoint)
             ? cast<arith::ConstantIntOp>(*it)
             : builder.createOrFold<arith::ConstantIntOp>(loc, val, bitWidth);
}

std::optional<tt::MakeTensorPtrOp> findDefiningMakeTensorPtrOp(Value val) {
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    Operation *parentOp = arg.getParentBlock()->getParentOp();

    Value loopArg;
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
      loopArg = forOp.getInitArgs()[arg.getArgNumber() - 1];
    else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      loopArg = whileOp.getInits()[arg.getArgNumber()];
    else
      llvm_unreachable("Unexpected parent operator");

    return findDefiningMakeTensorPtrOp(loopArg);
  }

  if (auto poisonOp = val.getDefiningOp<ub::PoisonOp>())
    return std::nullopt;
  if (auto callOp = val.getDefiningOp<tt::CallOp>())
    return std::nullopt;
  if (auto advanceOp = val.getDefiningOp<tt::AdvanceOp>())
    return findDefiningMakeTensorPtrOp(advanceOp.getPtr());
  if (auto makePtrOp = val.getDefiningOp<tt::MakeTensorPtrOp>())
    return makePtrOp;
  if (auto opRes = dyn_cast<OpResult>(val)) {
    Operation *defOp = opRes.getOwner();
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(defOp))
      return findDefiningMakeTensorPtrOp(
          loopOp.getYieldedValues()[opRes.getResultNumber()]);
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      // Give up if the 2 possible definitions aren't the same.
      Region &thenRgn = ifOp.getThenRegion();
      Region &elseRgn = ifOp.getElseRegion();
      assert(thenRgn.hasOneBlock() && elseRgn.hasOneBlock() &&
             "Expecting single blocks on both the 'then' and 'else' regions");
      auto thenYieldOp =
               cast<scf::YieldOp>(thenRgn.getBlocks().front().getTerminator()),
           elseYieldOp =
               cast<scf::YieldOp>(elseRgn.getBlocks().front().getTerminator());
      Value thenVal = thenYieldOp->getOperand(opRes.getResultNumber()),
            elseVal = elseYieldOp->getOperand(opRes.getResultNumber());
      std::optional<tt::MakeTensorPtrOp> thenDef = findDefiningMakeTensorPtrOp(
                                             thenVal),
                                         elseDef = findDefiningMakeTensorPtrOp(
                                             elseVal);
      if (!thenDef || !elseDef || *thenDef != *elseDef)
        return std::nullopt;
      return thenDef;
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      // Give up if the 2 possible definitions aren't the same.
      Value trueVal = selectOp.getTrueValue(),
            falseVal = selectOp.getFalseValue();
      std::optional<tt::MakeTensorPtrOp> trueDef = findDefiningMakeTensorPtrOp(
                                             trueVal),
                                         falseDef = findDefiningMakeTensorPtrOp(
                                             falseVal);
      if (!trueDef || !falseDef || *trueDef != *falseDef)
        return std::nullopt;
      return trueDef;
    }

    llvm::errs() << "defOp: " << *defOp << "\n";
    assert(false && "unhandled operation");
  }

  return std::nullopt;
}

std::optional<int64_t> getFoldedConstantValue(Operation *op) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(results)))
    return std::nullopt;

  // If fold succeeded but `results` is empty, we give a second try, after the
  // operands have been switched during the first call to `fold()`.
  if (results.empty()) {
    if (failed(op->fold(results)))
      return std::nullopt;
  }

  if (results.size() != 1)
    return std::nullopt;

  return getConstantIntValue(results[0]);
}

bool isConstant(Value val, int64_t expected) {
  if (auto defOp = val.getDefiningOp())
    return (getFoldedConstantValue(defOp) == expected);
  return false;
}

Value getFinalValue(Value value) {
  assert(value && "Expecting a valid value");
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // Look up init values outside the loop.
    auto blockArg = cast<BlockArgument>(value);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar())
        return value;

      int numIVs = forOp.getNumInductionVars();
      int initArgIdx = blockArg.getArgNumber() - numIVs;
      auto initArgs = forOp.getInitArgs();
      assert(initArgIdx >= 0 && initArgIdx < initArgs.size() &&
             "Unexpected 'initArgIdx' value");
      return getFinalValue(initArgs[initArgIdx]);
    }

    return value;
  }

  if (isa<tt::ExpandDimsOp, tt::BroadcastOp, tt::SplatOp, arith::IndexCastOp>(
          defOp))
    return getFinalValue(defOp->getOperand(0));

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    if (isConstant(addOp.getLhs(), 0))
      return getFinalValue(addOp.getRhs());
    if (isConstant(addOp.getRhs(), 0))
      return getFinalValue(addOp.getLhs());
    return addOp.getResult();
  }

  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    if (isConstant(subOp.getRhs(), 0))
      return getFinalValue(subOp.getLhs());
    return subOp.getResult();
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (isConstant(mulOp.getLhs(), 1) || isConstant(mulOp.getRhs(), 0))
      return getFinalValue(mulOp.getRhs());
    if (isConstant(mulOp.getRhs(), 1) || isConstant(mulOp.getLhs(), 0))
      return getFinalValue(mulOp.getLhs());
    return mulOp.getResult();
  }

  if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
    if (isConstant(divOp.getRhs(), 1) || isConstant(divOp.getLhs(), 0))
      return getFinalValue(divOp.getLhs());
    return divOp.getResult();
  }

  return value;
}

void eraseOperations(SmallPtrSetImpl<Operation *> &operations) {
  bool erasedOperation;
  do {
    erasedOperation = false;
    SmallPtrSet<Operation *, 8> erased;
    for (Operation *op : operations) {
      if (!op->getUsers().empty() || !op->getRegions().empty())
        continue;

      erased.insert(op);
      op->erase();
      erasedOperation = true;
    }
    operations.remove_if([&](Operation *op) { return erased.contains(op); });
  } while (erasedOperation);

  // Remove operations that contain a region.
  for (Operation *op : operations) {
    if (!op->getUsers().empty())
      continue;
    op->erase();
  }
}

} // namespace mlir::triton::intel
