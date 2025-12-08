#include "intel/include/Utils/SpecConst.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir::triton::intel {

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

static void collectRootBlockArgsProjected(Value v,
                                          SmallVector<int64_t, 4> idxPath,
                                          llvm::SetVector<BlockArgument> &roots,
                                          DenseSet<Value> &vis);

static void traceNonEntryBlockArgToIncomingValues(
    BlockArgument ba, SmallVector<int64_t, 4> idxPath,
    llvm::SetVector<BlockArgument> &roots, DenseSet<Value> &vis) {

  Block *dst = ba.getOwner();
  if (dst->hasNoPredecessors()) {
    // Entry block argument (true root)
    roots.insert(ba);
    return;
  }

  unsigned argIdx = ba.getArgNumber();

  for (Block *pred : dst->getPredecessors()) {
    Operation *term = pred->getTerminator();

    auto handleIncomingRange = [&](ValueRange ops) {
      if (argIdx < ops.size())
        collectRootBlockArgsProjected(ops[argIdx], idxPath, roots, vis);
    };

    if (auto br = dyn_cast<LLVM::BrOp>(term)) {
      if (br.getDest() == dst)
        handleIncomingRange(br.getDestOperands());
      continue;
    }

    if (auto cbr = dyn_cast<LLVM::CondBrOp>(term)) {
      if (cbr.getTrueDest() == dst)
        handleIncomingRange(cbr.getTrueDestOperands());
      if (cbr.getFalseDest() == dst)
        handleIncomingRange(cbr.getFalseDestOperands());
      continue;
    }

    // Unknown terminator: conservatively bail to “root is this ba”
    roots.insert(ba);
  }
}

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
    if (!ba.getOwner()->hasNoPredecessors()) {
      traceNonEntryBlockArgToIncomingValues(ba, idxPath, roots, vis);
      return;
    }
    roots.insert(ba);
    return;
  }

  if (isConstantLike(v))
    return;

  Operation *def = v.getDefiningOp();
  if (!def)
    return;

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

static FailureOr<BlockArgument> findSingleRootArgProjected(Value baseValue,
                                                           Location /*loc*/,
                                                           Operation *diagOp) {
  assert(diagOp && "diagOp must not be null");

  llvm::SetVector<BlockArgument> roots;
  DenseSet<Value> vis;
  collectRootBlockArgsProjected(baseValue, /*idxPath=*/{}, roots, vis);

  if (roots.size() == 1)
    return roots[0];

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "baseValue depends on " << roots.size()
     << " block arguments; expected 1.\n";
  os << "Root args:\n";
  for (BlockArgument ba : roots) {
    os << "  ";
    printBlockArgInfo(ba, os);
    os << "\n";
  }
  os.flush();

  diagOp->emitRemark() << os.str();
  return failure();
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

static FailureOr<Value> cloneWithBackwardSlice(Value baseValue,
                                               BlockArgument leaf,
                                               Value specArg, Location loc,
                                               PatternRewriter &rewriter,
                                               Operation *diagOp) {
  assert(diagOp && "diagOp must not be null");

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
      os.flush();

      diagOp->emitRemark() << os.str();
      return failure();
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
  if (failed(getBackwardSlice(baseValue, &slice, opts))) {
    diagOp->emitRemark() << "getBackwardSlice(baseValue) failed";
    return failure();
  }

  IRMapping mapping;
  mapping.map(leaf, specLeaf);

  SmallVector<Operation *> pending(slice.begin(), slice.end());

  auto valueReady = [&](Value v) -> bool {
    if (mapping.contains(v))
      return true;

    if (isa<BlockArgument>(v))
      return true;

    Operation *def = v.getDefiningOp();
    if (!def)
      return true;

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
      diagOp->emitRemark() << "Could not clone slice in dependency order "
                              "(cycle or unmapped leaf-alias).";
      return failure();
    }
  }

  return mapping.lookupOrDefault(baseValue);
}

static FailureOr<int32_t> addSpecConstArgIndexToModule(ModuleOp module,
                                                       int32_t argNo,
                                                       Operation *diagOp,
                                                       Location /*loc*/) {
  assert(diagOp && "diagOp must not be null");

  MLIRContext *ctx = module.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  int32_t count = 0;
  if (auto cAttr = module->getAttrOfType<IntegerAttr>("ttig.spec_const_count"))
    count = (int32_t)cAttr.getInt();

  for (int32_t i = 0; i < count; ++i) {
    std::string key = "ttig.spec_const_" + std::to_string(i);
    auto a = module->getAttrOfType<IntegerAttr>(key);
    if (!a) {
      diagOp->emitRemark() << "Missing expected module attr '" << key
                           << "' while ttig.spec_const_count=" << count;
      return failure();
    }
    if ((int32_t)a.getInt() == argNo)
      return i;
  }

  std::string newKey = "ttig.spec_const_" + std::to_string(count);
  module->setAttr(newKey, IntegerAttr::get(i32Ty, argNo));
  module->setAttr("ttig.spec_const_count", IntegerAttr::get(i32Ty, count + 1));
  return count;
}

static FailureOr<std::pair<BlockArgument, int32_t>>
markRootArgAsSpecConst(Value resultVal, ModuleOp module, Location loc,
                       Operation *diagOp) {
  assert(diagOp && "diagOp must not be null");

  auto leafOr = findSingleRootArgProjected(resultVal, loc, diagOp);
  if (failed(leafOr))
    return failure();
  BlockArgument leaf = *leafOr;

  auto idxOr =
      addSpecConstArgIndexToModule(module, leaf.getArgNumber(), diagOp, loc);
  if (failed(idxOr))
    return failure();

  // Preserve your current behavior: return arg number (not the deduped slot).
  int32_t arg = leaf.getArgNumber();
  return std::make_pair(leaf, arg);
}

FailureOr<Value> buildSpecConstBasedValue(Value baseValue, Operation *anchorOp,
                                          Location loc,
                                          PatternRewriter &rewriter) {
  assert(anchorOp && "buildSpecConstBasedValue: anchorOp must not be null");

  ModuleOp module = anchorOp->getParentOfType<ModuleOp>();
  if (!module) {
    anchorOp->emitRemark()
        << "buildSpecConstBasedValue: anchorOp has no ModuleOp";
    return failure();
  }

  Operation *diagOp = anchorOp;

  auto mr = markRootArgAsSpecConst(baseValue, module, loc, diagOp);
  if (failed(mr))
    return failure();
  auto [leaf, specConstIndex] = *mr;

  MLIRContext *ctx = rewriter.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, ArrayRef<Type>{i32Ty, i32Ty},
                                          /*isVarArg=*/false);

  LLVM::LLVMFuncOp specFn =
      module.lookupSymbol<LLVM::LLVMFuncOp>("__spirv_SpecConstant");
  if (!specFn) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    ImplicitLocOpBuilder ib(loc, rewriter);
    specFn = LLVM::LLVMFuncOp::create(ib, "__spirv_SpecConstant", fnTy);
  }

  Value specIdVal = LLVM::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(specConstIndex));

  Value defaultValue = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                rewriter.getI32IntegerAttr(0));

  auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                   SymbolRefAttr::get(specFn),
                                   ValueRange{specIdVal, defaultValue});

  Value specArg = call.getResult();

  return cloneWithBackwardSlice(baseValue, leaf, specArg, loc, rewriter,
                                diagOp);
}

} // namespace mlir::triton::intel
