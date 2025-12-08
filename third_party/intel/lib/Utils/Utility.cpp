#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
namespace tt = mlir::triton;

static std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
  return std::nullopt;
}

static bool isPrefix(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() > b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i])
      return false;
  return true;
}

static Value traceAggregateElement(Value v, SmallVectorImpl<int64_t> &idxPath) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto ex = dyn_cast<LLVM::ExtractValueOp>(def)) {
      Value container = ex.getContainer();

      SmallVector<int64_t, 4> exIdx;
      for (auto a : ex.getPosition())
        exIdx.push_back(a);

      // idxPath = exIdx + idxPath
      SmallVector<int64_t, 4> newPath;
      newPath.append(exIdx.begin(), exIdx.end());
      newPath.append(idxPath.begin(), idxPath.end());
      idxPath.swap(newPath);

      v = container;
      continue;
    }

    if (auto ins = dyn_cast<LLVM::InsertValueOp>(def)) {
      Value container = ins.getContainer();
      Value inserted = ins.getValue();

      SmallVector<int64_t, 4> insIdx;
      for (auto a : ins.getPosition())
        insIdx.push_back(a);

      if (isPrefix(insIdx, idxPath)) {
        // This insert overwrites the element (or a parent aggregate that
        // contains it).
        if (insIdx.size() == idxPath.size()) {
          v = inserted;
          idxPath.clear();
        } else {
          SmallVector<int64_t, 4> rest(idxPath.begin() + insIdx.size(),
                                       idxPath.end());
          v = inserted;
          idxPath.swap(rest);
        }
        continue;
      }

      // Insert to some other field: tracked field comes from container.
      v = container;
      continue;
    }

    break;
  }
  return v;
}

static bool isCastLike(Operation *op) {
  return isa<LLVM::TruncOp, LLVM::SExtOp, LLVM::ZExtOp, LLVM::BitcastOp,
             arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op);
}

static bool isConstantLike(Value v) { return matchPattern(v, m_Constant()); }

static void printBlockArgInfo(raw_ostream &os, BlockArgument ba) {
  os << "  - arg#" << ba.getArgNumber() << " : ";
  ba.print(os);
  os << " : " << ba.getType() << "\n";

  if (auto func =
          ba.getOwner()->getParentOp()->getParentOfType<LLVM::LLVMFuncOp>())
    os << "      in function: @" << func.getName() << "\n";
}

// Recursive collector.
// `idxPath` is the projection path we are tracking within aggregates.
static void collectRootBlockArgsProjected(Value v,
                                          SmallVector<int64_t, 4> idxPath,
                                          llvm::SetVector<BlockArgument> &roots,
                                          DenseSet<Value> &vis) {
  // Follow through insert/extract chains while keeping only the tracked field.
  v = traceAggregateElement(v, idxPath);

  if (!v || vis.contains(v))
    return;
  vis.insert(v);

  if (auto ba = dyn_cast<BlockArgument>(v)) {
    roots.insert(ba);
    return;
  }

  if (isConstantLike(v))
    return;

  Operation *def = v.getDefiningOp();
  if (!def)
    return;

  // Casts: keep projection path as-is.
  if (isCastLike(def)) {
    collectRootBlockArgsProjected(def->getOperand(0), idxPath, roots, vis);
    return;
  }

  // If we are still tracking an aggregate field here, but the defining op is
  // not insert/extract, we can’t safely “project” further without
  // special-casing. In practice for Triton block ptr packing, insert/extract
  // are what matters. Conservatively, just walk ALL operands (still works if
  // only one real arg).
  if (!idxPath.empty()) {
    for (Value opnd : def->getOperands())
      collectRootBlockArgsProjected(opnd, /*idxPath=*/{}, roots, vis);
    return;
  }

  // Common integer ops / generic fallback: walk operands.
  // (This naturally handles mixed add/mul/sub/and/or/shifts, etc.)
  for (Value opnd : def->getOperands())
    collectRootBlockArgsProjected(opnd, /*idxPath=*/{}, roots, vis);
}

static BlockArgument findSingleRootArgProjectedOrDie(Value pitch) {
  llvm::SetVector<BlockArgument> roots;
  DenseSet<Value> vis;
  collectRootBlockArgsProjected(pitch, /*idxPath=*/{}, roots, vis);

  if (roots.size() == 1)
    return roots[0];

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "pitch depends on " << roots.size()
     << " block arguments; expected 1.\n";
  os << "Root args:\n";
  for (BlockArgument ba : roots)
    printBlockArgInfo(os, ba);

  llvm::report_fatal_error(llvm::Twine(os.str()));
}

namespace mlir::triton::intel {

static void printBlockArgInfo(mlir::BlockArgument ba, llvm::raw_ostream &os) {
  os << "arg#" << ba.getArgNumber();

  if (auto llvmFn = ba.getOwner()
                        ->getParentOp()
                        ->getParentOfType<mlir::LLVM::LLVMFuncOp>()) {
    os << " (llvm.func @" << llvmFn.getName() << ")";
  } else if (auto fn = ba.getOwner()
                           ->getParentOp()
                           ->getParentOfType<mlir::func::FuncOp>()) {
    os << " (func @" << fn.getName() << ")";
  }

  if (auto *ownerOp = ba.getOwner()->getParentOp()) {
    if (auto attrDict = ownerOp->getAttrDictionary()) {
      (void)attrDict;
    }
  }

  os << " : " << ba.getType();
  os << " in block @" << (const void *)ba.getOwner();
}

static Value skipCasts(Value v) {
  Operation *def = v.getDefiningOp();
  if (def &&
      isa<LLVM::TruncOp, LLVM::SExtOp, LLVM::ZExtOp, LLVM::BitcastOp>(def))
    return def->getOperand(0);
  return v;
}

static Value castTo(Value v, Type dstTy, Location loc,
                    PatternRewriter &rewriter) {
  if (v.getType() == dstTy)
    return v;

  auto srcInt = dyn_cast<IntegerType>(v.getType());
  auto dstInt = dyn_cast<IntegerType>(dstTy);

  if (srcInt && dstInt) {
    unsigned s = srcInt.getWidth();
    unsigned d = dstInt.getWidth();
    if (s < d)
      return LLVM::ZExtOp::create(rewriter, loc, dstTy, v);
    if (s > d)
      return LLVM::TruncOp::create(rewriter, loc, dstTy, v);
    return v;
  }

  // conservative fallback
  return LLVM::BitcastOp::create(rewriter, loc, dstTy, v);
}

static BlockArgument findSingleRootArg(Value pitchVal) {
  Value v = skipCasts(pitchVal);
  if (auto ba = dyn_cast<BlockArgument>(v))
    return ba;

  Operation *def = pitchVal.getDefiningOp();
  assert(def && "pitchVal must be defined by an op for this helper");

  llvm::SetVector<Operation *> slice;
  BackwardSliceOptions opts;
  opts.inclusive = true;
  opts.omitBlockArguments = true;
  (void)mlir::getBackwardSlice(def, &slice, opts);

  BlockArgument only;
  for (Operation *op : slice) {
    for (Value operand : op->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(skipCasts(operand))) {
        if (!only)
          only = ba;
        else if (only != ba) {
          std::string msg;
          llvm::raw_string_ostream os(msg);
          os << "pitch depends on >1 block argument; pick one explicitly.\n";
          os << "  first  = ";
          printBlockArgInfo(only, os);
          os << "\n";
          os << "  second = ";
          printBlockArgInfo(ba, os);
          os << "\n";
          os.flush();
          llvm::report_fatal_error(msg.c_str());
        }
      }
    }
  }
  if (!only)
    llvm::report_fatal_error("could not find a root block argument for pitch");
  return only;
}

Value cloneWithBackwardSlice(Value pitchVal, BlockArgument leaf, Value specArg,
                             Location loc, PatternRewriter &rewriter) {
  Value specLeaf = specArg;
  if (specLeaf.getType() != leaf.getType()) {
    auto dstITy = dyn_cast<IntegerType>(leaf.getType());
    auto srcITy = dyn_cast<IntegerType>(specLeaf.getType());

    if (!dstITy || !srcITy) {
      std::string msg;
      llvm::raw_string_ostream os(msg);

      os << "Need integer types for specArg<->leaf cast\n";

      os << "  leaf arg# " << leaf.getArgNumber() << "\n";
      os << "  leaf type: ";
      leaf.getType().print(os);
      os << "\n";

      os << "  specArg type: ";
      specLeaf.getType().print(os);
      os << "\n";

      os << "  leaf value: ";
      leaf.print(os);
      os << "\n";

      if (auto llvmFn = leaf.getOwner()
                            ->getParentOp()
                            ->getParentOfType<mlir::LLVM::LLVMFuncOp>()) {
        os << "  in llvm.func @" << llvmFn.getName() << "\n";
      } else if (auto fn = leaf.getOwner()
                               ->getParentOp()
                               ->getParentOfType<mlir::func::FuncOp>()) {
        os << "  in func @" << fn.getName() << "\n";
      }

      os << "  leaf type class: " << leaf.getType().getDialect().getNamespace()
         << "\n";
      os << "  specArg type class: "
         << specLeaf.getType().getDialect().getNamespace() << "\n";

      os.flush();
      llvm::report_fatal_error(llvm::Twine(os.str()));
    }

    unsigned sw = srcITy.getWidth();
    unsigned dw = dstITy.getWidth();
    if (sw < dw)
      specLeaf = LLVM::ZExtOp::create(rewriter, loc, leaf.getType(), specLeaf);
    else if (sw > dw)
      specLeaf = LLVM::TruncOp::create(rewriter, loc, leaf.getType(), specLeaf);
  }

  llvm::SetVector<Operation *> slice;
  BackwardSliceOptions opts;
  opts.inclusive = true;
  if (failed(getBackwardSlice(pitchVal, &slice, opts)))
    llvm::report_fatal_error("getBackwardSlice(pitchVal) failed");

  bool found = false;
  for (Operation *op : slice)
    for (Value o : op->getOperands())
      if (o == leaf)
        found = true;

  if (!found)
    llvm::errs() << "WARNING: slice never uses `leaf` directly; mapping "
                    "leaf->specLeaf won't trigger.\n";

  IRMapping mapping;
  mapping.map(leaf, specLeaf);

  SmallVector<Operation *> pending(slice.begin(), slice.end());

  auto valueReady = [&](Value v) -> bool {
    if (mapping.contains(v))
      return true;

    // Block arguments / constants / values defined outside the slice are
    // "ready".
    if (isa<BlockArgument>(v))
      return true;

    Operation *def = v.getDefiningOp();
    if (!def)
      return true;

    // If the def op is in the slice, v is NOT ready until that def was cloned.
    return !slice.contains(def);
  };

  while (!pending.empty()) {
    bool progress = false;

    for (auto it = pending.begin(); it != pending.end();) {
      Operation *op = *it;

      bool ready = true;
      for (Value operand : op->getOperands()) {
        if (!valueReady(operand)) {
          ready = false;
          break;
        }
      }

      if (!ready) {
        ++it;
        continue;
      }

      rewriter.clone(*op, mapping);
      it = pending.erase(it);
      progress = true;
    }

    if (!progress) {
      // Usually means: ordering cycle in pending
      llvm::report_fatal_error("Could not clone slice in dependency order "
                               "(cycle or unmapped leaf-alias).");
    }
  }

  return mapping.lookupOrDefault(pitchVal);
}

static int32_t addSpecConstArgIndexToModule(ModuleOp module, int32_t argNo) {
  MLIRContext *ctx = module.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  // Read count (default 0)
  int32_t count = 0;
  if (auto cAttr = module->getAttrOfType<IntegerAttr>("ttig.spec_const_count"))
    count = (int32_t)cAttr.getInt();

  // Check existing entries for dedup; if found, return.
  for (int32_t i = 0; i < count; ++i) {
    std::string key = "ttig.spec_const_" + std::to_string(i);
    auto a = module->getAttrOfType<IntegerAttr>(key);
    if (!a) {
      // If count says key exists but it doesn't, treat as an error.
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Missing expected module attr '" << key
         << "' while ttig.spec_const_count=" << count;
      os.flush();
      llvm::report_fatal_error(llvm::Twine(os.str()));
    }
    if ((int32_t)a.getInt() == argNo)
      return i; // already recorded
  }

  // Append new entry at slot = count
  std::string newKey = "ttig.spec_const_" + std::to_string(count);
  module->setAttr(newKey, IntegerAttr::get(i32Ty, argNo));

  // Bump count
  module->setAttr("ttig.spec_const_count", IntegerAttr::get(i32Ty, count + 1));

  return count;
}

std::pair<BlockArgument, int32_t> markRootArgAsSpecConst(Value resultVal) {
  BlockArgument leaf = findSingleRootArgProjectedOrDie(resultVal);

  Operation *funcLikeOp = leaf.getOwner()->getParentOp(); // any op in module
  int32_t specConstIndex = addSpecConstArgIndexToModule(
      funcLikeOp->getParentOfType<ModuleOp>(), leaf.getArgNumber());

  // TODO: in order to support tensor desc properly we have to change mapping
  int32_t arg = leaf.getArgNumber();
  return {leaf, arg};
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
