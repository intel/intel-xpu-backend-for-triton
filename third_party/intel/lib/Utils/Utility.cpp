#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

  if (auto advanceOp = val.getDefiningOp<tt::AdvanceOp>())
    return findDefiningMakeTensorPtrOp(advanceOp.getPtr());
  if (auto makePtrOp = val.getDefiningOp<tt::MakeTensorPtrOp>())
    return makePtrOp;
  if (auto opRes = dyn_cast<OpResult>(val)) {
    Operation *defOp = opRes.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      Value val = forOp.getYieldedValues()[opRes.getResultNumber()];
      return findDefiningMakeTensorPtrOp(val);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      Value val = whileOp.getYieldedValues()[opRes.getResultNumber()];
      return findDefiningMakeTensorPtrOp(val);
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      // Give up if the 2 possible definitions aren't the same.
      Value trueVal = selectOp.getTrueValue(),
            falseVal = selectOp.getFalseValue();
      std::optional<tt::MakeTensorPtrOp> trueDef =
          findDefiningMakeTensorPtrOp(trueVal);
      std::optional<tt::MakeTensorPtrOp> falseDef =
          findDefiningMakeTensorPtrOp(falseVal);
      if (!trueDef || !falseDef || *trueDef != *falseDef)
        return std::nullopt;
      return trueDef;
    }

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

  std::optional<int64_t> intAttr = getIntAttr(results[0]);
  if (intAttr.has_value())
    return intAttr.value();

  auto val = cast<Value>(results[0]);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  return getIntAttr(constOp.getValue());
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
    // look init values outside the loop
    BlockArgument blockArg = cast<BlockArgument>(value);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
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

} // namespace mlir::triton::intel
