#ifndef TRITON_INTEL_UTILS_UTILITY_H
#define TRITON_INTEL_UTILS_UTILITY_H

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
class MakeTensorPtrOp;
class MakeTensorDescOp;
} // namespace triton

class FunctionOpInterface;
class LoopLikeOpInterface;
} // namespace mlir

namespace mlir::triton::intel {

// Lookup for a integer constant with the given value and bitwidth in the
// current block (before the builder insertion point). Return it if found,
// otherwise create a new one.
Value findOrCreateIntConstant(Location loc, int val, unsigned bitWidth,
                              OpBuilder &builder);

// This function folds the `v` value and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Value v, int depth = 8);

// Return true if the `val` value is a constant containing a value equal to
// expected.
bool isConstant(Value val, int64_t expected);

Value getFinalValue(Value value);

// Erase the operations in \p operations.
void eraseOperations(SmallPtrSetImpl<Operation *> &operations);

// Find the defining operation of type `OpTy` for the given value.
// Note: traverses block arguments and loop yields, etc...
template <typename OpTy,
          typename = std::enable_if<llvm::is_one_of<
              OpTy, triton::MakeTensorPtrOp, triton::MakeTensorDescOp>::value>>
std::optional<OpTy> findDefiningOpOfType(Value val) {
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    Operation *parentOp = arg.getParentBlock()->getParentOp();
    if (!parentOp || isa<FunctionOpInterface>(parentOp))
      return std::nullopt;

    Value loopArg;
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
      loopArg = forOp.getInitArgs()[arg.getArgNumber() - 1];
    else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      loopArg = whileOp.getInits()[arg.getArgNumber()];
    else
      llvm_unreachable("Unexpected parent operator");

    return findDefiningOpOfType<OpTy>(loopArg);
  }

  if (auto poisonOp = val.getDefiningOp<ub::PoisonOp>())
    return std::nullopt;
  if (auto callOp = val.getDefiningOp<triton::CallOp>())
    return std::nullopt;
  if (auto advanceOp = val.getDefiningOp<triton::AdvanceOp>())
    return findDefiningOpOfType<OpTy>(advanceOp.getPtr());
  if (auto makePtrOp = val.getDefiningOp<OpTy>())
    return makePtrOp;
  if (auto opRes = dyn_cast<OpResult>(val)) {
    Operation *defOp = opRes.getOwner();
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(defOp))
      return findDefiningOpOfType<OpTy>(
          loopOp.getYieldedValues()[opRes.getResultNumber()]);
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      // Give up if the 2 possible definitions aren't the same.
      Region &thenRgn = ifOp.getThenRegion();
      Region &elseRgn = ifOp.getElseRegion();
      if (thenRgn.empty() || elseRgn.empty())
        return std::nullopt;
      assert(thenRgn.hasOneBlock() && elseRgn.hasOneBlock() &&
             "Expecting single blocks on both the 'then' and 'else' regions");
      auto thenYieldOp =
               cast<scf::YieldOp>(thenRgn.getBlocks().front().getTerminator()),
           elseYieldOp =
               cast<scf::YieldOp>(elseRgn.getBlocks().front().getTerminator());
      Value thenVal = thenYieldOp->getOperand(opRes.getResultNumber()),
            elseVal = elseYieldOp->getOperand(opRes.getResultNumber());
      std::optional<OpTy> thenDef = findDefiningOpOfType<OpTy>(thenVal),
                          elseDef = findDefiningOpOfType<OpTy>(elseVal);
      if (!thenDef || !elseDef || *thenDef != *elseDef)
        return std::nullopt;
      return thenDef;
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      // Give up if the 2 possible definitions aren't the same.
      Value trueVal = selectOp.getTrueValue(),
            falseVal = selectOp.getFalseValue();
      std::optional<OpTy> trueDef = findDefiningOpOfType<OpTy>(trueVal),
                          falseDef = findDefiningOpOfType<OpTy>(falseVal);
      if (!trueDef || !falseDef || *trueDef != *falseDef)
        return std::nullopt;
      return trueDef;
    }

    llvm::errs() << "defOp: " << *defOp << "\n";
    assert(false && "unhandled operation");
  }

  return std::nullopt;
}

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_UTILITY_H
