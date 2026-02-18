#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static Value skipCasts(Value v) {
  Operation *def = v.getDefiningOp();
  if (def &&
      isa<LLVM::TruncOp, LLVM::SExtOp, LLVM::ZExtOp, LLVM::BitcastOp>(def))
    return def->getOperand(0);
  return v;
}

static Value foldValue(Value v) {
  if (Operation *def = v.getDefiningOp()) {
    SmallVector<OpFoldResult> results;

    if (failed(def->fold(results)))
      return v;

    // If fold succeeded but `results` is empty, we give a second try, after the
    // operands have been switched during the first call to `fold()`.
    if (results.empty()) {
      if (failed(def->fold(results)))
        return v;
    }

    if (results.size() == 1) {
      if (auto val = dyn_cast_or_null<Value>(results[0]))
        return val;
    }
  }
  return v;
}

std::optional<int64_t> getFoldedConstantValue(Value v, int depth) {
  for (int i = 0; i < depth; ++i) {
    if (auto res = getConstantIntValue(v))
      return res;

    Value newV = skipCasts(v);
    newV = foldValue(newV);

    if (newV == v)
      break;

    v = newV;
  }

  return std::nullopt;
}

bool isConstant(Value val, int64_t expected) {
  return (getFoldedConstantValue(val) == expected);
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

  if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
    return getFinalValue(extOp.getIn());
  if (auto extOp = dyn_cast<arith::ExtUIOp>(defOp))
    return getFinalValue(extOp.getIn());

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
