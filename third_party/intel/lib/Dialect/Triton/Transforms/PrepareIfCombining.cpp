//===----------------------------------------------------------------------===//
// This pass enables combining of scf.if operations that share the same
// condition by moving intervening operations out of the way.
//
// Example:
//   %0 = ttg.local_load %smem : ...
//   %1 = scf.if %cond -> tensor<...> { ... }
//   %2 = tt.trans %0 : ...  // <-- blocking combining if ops
//   %3 = scf.if %cond -> tensor<...> { ... }
//
// After this pass, %2 is moved before %1, making %1 and %3 adjacent so that
// they can be combined by the canonicalizer.
//
// Ops that cannot move above the first if (they use its results or touch
// memory) are copied into both of its branches instead. See Passes.td for the
// conditions.
//===----------------------------------------------------------------------===//
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {

#define GEN_PASS_DEF_TRITONINTELGPUPREPAREIFCOMBINING
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir::triton::intel

namespace {

constexpr unsigned kMaxConditionDepth = 8;

/// True if `a` and `b` are the same value or are computed by identical
/// side-effect-free expressions.
static bool sameCondition(Value a, Value b, unsigned depth = 0) {
  if (a == b)
    return true;
  auto resA = dyn_cast<OpResult>(a);
  auto resB = dyn_cast<OpResult>(b);
  if (!resA || !resB || resA.getResultNumber() != resB.getResultNumber() ||
      depth == kMaxConditionDepth)
    return false;
  Operation *defA = resA.getOwner();
  Operation *defB = resB.getOwner();
  if (!isMemoryEffectFree(defA) || !isMemoryEffectFree(defB) ||
      defA->getNumRegions() || defB->getNumRegions())
    return false;
  return OperationEquivalence::isEquivalentTo(
      defA, defB,
      [&](Value lhs, Value rhs) {
        return success(sameCondition(lhs, rhs, depth + 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

/// True if `cond` has the same value in every thread of the program: it is
/// built by arith/math ops from constants, program ids and arguments of the
/// public kernel entry. Effect-free is not enough (`ttg.warp_id`, inline asm),
/// and copied collectives such as reductions lower to work-group barriers.
static bool isUniformCondition(Value cond, tt::FuncOp funcOp) {
  SmallVector<Value> worklist{cond};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      // Arguments of a private helper may come from a per-warp caller.
      if (!funcOp.isPublic() || arg.getOwner() != &funcOp.getBody().front())
        return false;
      continue;
    }
    Operation *def = value.getDefiningOp();
    if (isa<tt::GetProgramIdOp, tt::GetNumProgramsOp>(def))
      continue;
    StringRef dialect = def->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") || def->getNumRegions())
      return false;
    worklist.append(def->operand_begin(), def->operand_end());
  }
  return true;
}

/// True if `value` is defined before `head`, in its block or an enclosing one.
static bool isAvailableBefore(Value value, Operation *head) {
  Block *block = head->getBlock();
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getOwner() == block ||
           arg.getOwner()->findAncestorOpInBlock(*head);
  Operation *ancestor = block->findAncestorOpInBlock(*value.getDefiningOp());
  return !ancestor || ancestor->isBeforeInBlock(head);
}

/// Values that may leave an scf.if as results. Descriptors and block pointers
/// are excluded: their users trace them to a single definition.
static bool isEscapableType(Type type) {
  if (auto ptr = dyn_cast<tt::PointerType>(type))
    return !isa<RankedTensorType>(ptr.getPointeeType());
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return type.isIntOrIndexOrFloat();
}

/// The first of the adjacent scf.if ops ending at `ifOp` that test its
/// condition.
static Operation *getRunHead(scf::IfOp ifOp) {
  Operation *head = ifOp;
  while (auto prev = dyn_cast_or_null<scf::IfOp>(head->getPrevNode())) {
    if (prev.getCondition() != ifOp.getCondition())
      break;
    head = prev;
  }
  return head;
}

/// Copies `ops` to the end of both branches of `ifOp` and erases them. Values
/// in `escaping` (used after `ops`) become extra results of a rebuilt `ifOp`.
static void sinkInto(scf::IfOp ifOp, ArrayRef<Operation *> ops,
                     ArrayRef<Value> escaping) {
  OpBuilder builder(ifOp);
  for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    auto yield = cast<scf::YieldOp>(region->front().getTerminator());
    IRMapping mapping;
    mapping.map(ifOp.getResults(), yield.getOperands());
    builder.setInsertionPoint(yield);
    for (Operation *op : ops)
      builder.clone(*op, mapping);
    for (Value value : escaping)
      yield.getResultsMutable().append(mapping.lookup(value));
  }

  if (!escaping.empty()) {
    // Same steps as replaceIfOpWithNewSignature (TritonGPU transforms).
    SmallVector<Type> types(ifOp.getResultTypes());
    for (Value value : escaping)
      types.push_back(value.getType());
    builder.setInsertionPoint(ifOp);
    auto result =
        scf::IfOp::create(builder, ifOp.getLoc(), types, ifOp.getCondition());
    result->setAttrs(ifOp->getAttrs());
    result.getThenRegion().takeBody(ifOp.getThenRegion());
    result.getElseRegion().takeBody(ifOp.getElseRegion());
    unsigned numOld = ifOp.getNumResults();
    ifOp.replaceAllUsesWith(result.getResults().take_front(numOld));
    for (auto [i, value] : llvm::enumerate(escaping))
      Value(value).replaceAllUsesWith(result.getResult(numOld + i));
    ifOp.erase();
  }
  for (Operation *op : llvm::reverse(ops))
    op->erase();
}

/// Makes `currentIf` and `nextIf` adjacent if they test the same condition.
/// Changes nothing unless that succeeds. `currentIf` may be replaced.
static void handleIfPair(scf::IfOp currentIf, scf::IfOp nextIf,
                         tt::FuncOp funcOp, unsigned maxClonedOps) {
  Value cond = currentIf.getCondition();
  if (!sameCondition(cond, nextIf.getCondition()))
    return;

  SmallVector<Operation *> between;
  for (Operation *op = currentIf->getNextNode(); op != nextIf.getOperation();
       op = op->getNextNode())
    between.push_back(op);

  // Ops that only compute nextIf's duplicate condition die once it uses
  // `cond`.
  DenseSet<Operation *> deadGuardOps;
  if (nextIf.getCondition() != cond) {
    for (Operation *op : llvm::reverse(between)) {
      if (op->getNumRegions() || !isMemoryEffectFree(op) || op->use_empty())
        continue;
      if (llvm::all_of(op->getUsers(), [&](Operation *user) {
            return user == nextIf.getOperation() || deadGuardOps.contains(user);
          }))
        deadGuardOps.insert(op);
    }
  }

  // Pure ops whose inputs, including those used inside their regions, exist
  // before the run head are hoisted; everything else is sunk into currentIf.
  Operation *head = getRunHead(currentIf);
  DenseSet<Operation *> hoisted;
  SmallVector<Operation *> toHoist, toSink;
  for (Operation *op : between) {
    if (deadGuardOps.contains(op))
      continue;
    SetVector<Value> used(op->operand_begin(), op->operand_end());
    getUsedValuesDefinedAbove(op->getRegions(), used);
    bool hoistable = isPure(op) && llvm::all_of(used, [&](Value value) {
                       Operation *def = value.getDefiningOp();
                       return (def && hoisted.contains(def)) ||
                              isAvailableBefore(value, head);
                     });
    if (hoistable) {
      hoisted.insert(op);
      toHoist.push_back(op);
    } else {
      toSink.push_back(op);
    }
  }

  SmallVector<Value> escaping;
  if (!toSink.empty()) {
    if (currentIf.getElseRegion().empty() || !isUniformCondition(cond, funcOp))
      return;
    DenseSet<Operation *> sinking(toSink.begin(), toSink.end());
    unsigned numOps = 0;
    for (Operation *op : toSink) {
      bool hasNestedIf = false;
      op->walk([&](Operation *inner) {
        ++numOps;
        hasNestedIf |= isa<scf::IfOp>(inner);
      });
      if (hasNestedIf)
        return;
      for (Value value : op->getResults()) {
        bool usedOutsideSink =
            llvm::any_of(value.getUsers(), [&](Operation *user) {
              return !sinking.contains(
                  op->getBlock()->findAncestorOpInBlock(*user));
            });
        if (!usedOutsideSink)
          continue;
        if (!isEscapableType(value.getType()))
          return;
        escaping.push_back(value);
      }
    }
    if (numOps > maxClonedOps)
      return;
  }

  if (nextIf.getCondition() != cond)
    nextIf.getConditionMutable().assign(cond);
  for (Operation *op : llvm::reverse(between))
    if (deadGuardOps.contains(op))
      op->erase();
  for (Operation *op : toHoist)
    op->moveBefore(head);
  if (!toSink.empty())
    sinkInto(currentIf, toSink, escaping);
}

struct PrepareIfCombiningPass
    : public mlir::triton::intel::impl::TritonIntelGPUPrepareIfCombiningBase<
          PrepareIfCombiningPass> {
  using Base::Base;

  void runOnOperation() override {
    triton::FuncOp funcOp = getOperation();
    llvm::MapVector<Block *, SmallVector<scf::IfOp>> ifsByBlock;
    funcOp.walk(
        [&](scf::IfOp ifOp) { ifsByBlock[ifOp->getBlock()].push_back(ifOp); });
    for (auto &[block, ifOps] : ifsByBlock)
      for (size_t i = 0; i + 1 < ifOps.size(); ++i)
        handleIfPair(ifOps[i], ifOps[i + 1], funcOp, maxClonedOps);
  }
};

} // namespace
