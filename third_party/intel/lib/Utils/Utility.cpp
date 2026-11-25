#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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

Value findOrCreateCastOp(Value val, Type targetType) {
  Location loc = val.getLoc();
  OpBuilder builder(val.getContext());

  auto isMatchingCastOp = [&](CastOpInterface castOp) {
    return castOp->getNumOperands() == 1 && castOp->getOperand(0) == val &&
           castOp->getNumResults() == 1 &&
           castOp->getResult(0).getType() == targetType;
  };

  if (Operation *defOp = val.getDefiningOp()) {
    Operation *nextOp = defOp->getNextNode();
    if (auto castOp = dyn_cast<CastOpInterface>(nextOp)) {
      if (isMatchingCastOp(castOp))
        return castOp->getResult(0);
    }

    builder.setInsertionPointAfter(defOp);
  } else {
    Value foundCast = nullptr;
    val.getParentBlock()->walk([&](Operation *op) {
      if (isa<arith::ConstantOp>(op))
        return WalkResult::advance();

      auto castOp = dyn_cast<CastOpInterface>(op);
      if (!castOp)
        return WalkResult::interrupt();

      if (isMatchingCastOp(castOp)) {
        foundCast = castOp->getResult(0);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (foundCast)
      return foundCast;

    builder.setInsertionPointToStart(val.getParentBlock());
  }

  return getValueOrCreateCastToIndexLike(builder, loc, targetType, val);
}

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

  if (isa<tt::ExpandDimsOp, tt::BroadcastOp, tt::SplatOp, arith::IndexCastOp,
          arith::ExtSIOp, arith::ExtUIOp>(defOp))
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

SmallVector<tt::MakeTensorDescOp> findAllMakeTensorDescOps(Value val) {
  llvm::SmallSetVector<tt::MakeTensorDescOp, 4> results;
  SmallPtrSet<Value, 8> visited;
  SmallVector<Value, 8> worklist;
  worklist.push_back(val);

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    if (auto arg = dyn_cast<BlockArgument>(cur)) {
      Operation *parentOp = arg.getParentBlock()->getParentOp();
      if (!parentOp || isa<FunctionOpInterface>(parentOp))
        return {};

      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        // The induction variable (argNumber == 0) is not traceable.
        if (arg == forOp.getInductionVar())
          return {};
        unsigned idx = arg.getArgNumber() - 1;
        worklist.push_back(forOp.getInitArgs()[idx]);
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        worklist.push_back(yieldOp->getOperand(idx));
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned idx = arg.getArgNumber();
        Block *beforeBlock = &whileOp.getBefore().front();
        Block *afterBlock = &whileOp.getAfter().front();
        auto condOp = cast<scf::ConditionOp>(beforeBlock->getTerminator());
        auto afterYieldOp = cast<scf::YieldOp>(afterBlock->getTerminator());

        if (arg.getParentBlock() == beforeBlock) {
          worklist.push_back(whileOp.getInits()[idx]);
          worklist.push_back(afterYieldOp->getOperand(idx));
        } else {
          worklist.push_back(condOp.getArgs()[idx]);
        }
        continue;
      }
      // Unknown parent op — cannot trace through.
      return {};
    }

    // PoisonOp is an uninitialized placeholder (e.g., in pipelined loops).
    // It is not a valid definition but does not invalidate other results.
    if (cur.getDefiningOp<ub::PoisonOp>())
      continue;
    if (cur.getDefiningOp<tt::CallOp>())
      return {};
    if (auto makeDescOp = cur.getDefiningOp<tt::MakeTensorDescOp>()) {
      results.insert(makeDescOp);
      continue;
    }
    if (auto opRes = dyn_cast<OpResult>(cur)) {
      Operation *defOp = opRes.getOwner();
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(defOp)) {
        worklist.push_back(loopOp.getYieldedValues()[opRes.getResultNumber()]);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        Region &thenRgn = ifOp.getThenRegion();
        Region &elseRgn = ifOp.getElseRegion();
        if (!thenRgn.empty()) {
          auto thenYieldOp =
              cast<scf::YieldOp>(thenRgn.getBlocks().front().getTerminator());
          worklist.push_back(thenYieldOp->getOperand(opRes.getResultNumber()));
        }
        if (!elseRgn.empty()) {
          auto elseYieldOp =
              cast<scf::YieldOp>(elseRgn.getBlocks().front().getTerminator());
          worklist.push_back(elseYieldOp->getOperand(opRes.getResultNumber()));
        }
        continue;
      }
      if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
        worklist.push_back(selectOp.getTrueValue());
        worklist.push_back(selectOp.getFalseValue());
        continue;
      }
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        if (castOp.getInputs().size() != 1)
          return {};
        worklist.push_back(castOp.getInputs()[0]);
        continue;
      }
      // Unknown op producing this value — cannot trace through.
      return {};
    }
  }

  return results.takeVector();
}

std::optional<tt::MakeTensorDescOp> findMakeTensorDescOp(Value val) {
  SmallVector<tt::MakeTensorDescOp> all = findAllMakeTensorDescOps(val);
  if (all.size() == 1)
    return all[0];
  return std::nullopt;
}

} // namespace mlir::triton::intel
