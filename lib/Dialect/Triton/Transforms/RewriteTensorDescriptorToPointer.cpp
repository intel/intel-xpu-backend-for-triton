#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREWRITETENSORDESCRIPTORTOPOINTER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

using namespace mlir;

using MakeDescSet = llvm::SmallSetVector<triton::MakeTensorDescOp, 4>;

/// Collect into `results` every `MakeTensorDescOp` reachable from `val`,
/// threading through scf iter-args/results, select, and casts. Returns false if
/// the walk hit a value it cannot see through (function arg, call, induction
/// var, unknown op) — i.e. `results` is only a *partial* answer, because some
/// descriptor we cannot name also flows into `val`. The walk always runs to
/// completion so `results` holds everything we did manage to see; callers that
/// only care about a fully-classified value use `findAllMakeTensorDescOps`.
static bool collectMakeTensorDescOps(Value val, MakeDescSet &results) {
  SmallPtrSet<Value, 8> visited;
  SmallVector<Value, 8> worklist{val};
  bool complete = true;

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    if (auto arg = dyn_cast<BlockArgument>(cur)) {
      Operation *parentOp = arg.getParentBlock()->getParentOp();
      if (!parentOp || isa<FunctionOpInterface>(parentOp)) {
        complete = false;
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        if (arg == forOp.getInductionVar()) {
          complete = false;
          continue;
        }
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
      complete = false; // unknown parent op
      continue;
    }

    // Poison placeholder (e.g. pipelined-loop init): skip, don't invalidate.
    if (cur.getDefiningOp<ub::PoisonOp>())
      continue;
    if (cur.getDefiningOp<triton::CallOp>()) {
      complete = false;
      continue;
    }
    if (auto makeDescOp = cur.getDefiningOp<triton::MakeTensorDescOp>()) {
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
        for (Region *rgn : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
          if (rgn->empty())
            continue;
          auto y = cast<scf::YieldOp>(rgn->front().getTerminator());
          worklist.push_back(y->getOperand(opRes.getResultNumber()));
        }
        continue;
      }
      if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
        worklist.push_back(selectOp.getTrueValue());
        worklist.push_back(selectOp.getFalseValue());
        continue;
      }
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        if (castOp.getInputs().size() != 1) {
          complete = false;
          continue;
        }
        worklist.push_back(castOp.getInputs()[0]);
        continue;
      }
      complete = false; // unknown defining op
      continue;
    }
    complete = false;
  }
  return complete;
}

/// Trace a tensor-descriptor Value back to its defining `MakeTensorDescOp`(s).
/// Returns empty ("cannot classify", handled conservatively by callers) for a
/// value we can't fully see through. Kept in sync with
/// third_party/intel/lib/Utils/Utility.cpp (upstream can't depend on it).
static SmallVector<triton::MakeTensorDescOp>
findAllMakeTensorDescOps(Value val) {
  MakeDescSet results;
  if (!collectMakeTensorDescOps(val, results))
    return {};
  return results.takeVector();
}

/// True if `value` would be loop-invariant *after* LICM, replicating
/// mlir::moveLoopInvariantCode's rule (pure op + recursively-invariant
/// operands). We can't just query LICM: this pass runs in make_ttir, before
/// triton-licm in make_ttgir, so hoistable temporaries are still in the loop.
/// Recursion stops at impure ops, so an in-loop `tt.load` base (the paged-KV
/// case) stays loop-varying. `memo` collapses diamond operand DAGs to O(N).
static bool isLoopInvariantAfterLICM(Value value, LoopLikeOpInterface loop,
                                     llvm::DenseMap<Value, bool> &memo) {
  if (loop.isDefinedOutsideOfLoop(value))
    return true;
  if (auto it = memo.find(value); it != memo.end())
    return it->second;
  Operation *def = value.getDefiningOp();
  bool invariant =
      def && isPure(def) && llvm::all_of(def->getOperands(), [&](Value v) {
        return isLoopInvariantAfterLICM(v, loop, memo);
      });
  memo[value] = invariant;
  return invariant;
}

/// A descriptor in a loop that LICM can't hoist is rebuilt each iteration,
/// paying a per-iteration tensormap_create on Hopper+ — the case we demote.
static bool isLoopRecreatedDescriptor(triton::MakeTensorDescOp desc) {
  auto loop = desc->getParentOfType<LoopLikeOpInterface>();
  if (!loop)
    return false; // out of loop -> hoistable / one-shot, keep it
  llvm::DenseMap<Value, bool> memo;
  return !llvm::all_of(desc->getOperands(), [&](Value v) {
    return isLoopInvariantAfterLICM(v, loop, memo);
  });
}

/// Grow `candidates` (the loop-recreated seeds) to a set the conversion can
/// rewrite consistently, or shrink it to nothing.
///
/// Demotion is all-or-nothing per *merge group*. The legality rules below judge
/// a `MakeTensorDescOp` by set membership but a consumer by "do all makes its
/// descriptor operand traces to belong to the set". When one descriptor value
/// merges several makes (scf.if / select / iter-arg), those two rules disagree
/// unless the makes are decided together: rewriting only the seed would leave
/// its consumer expecting a `!tt.tensordesc` that no longer exists, and the
/// type converter runs with `buildMaterializations = false`, so nothing bridges
/// the gap.
///
/// So: union every make that co-occurs in some descriptor value's provenance,
/// then decide each class as a whole.
///   - The class must stay on TD if it reaches a consumer this mode cannot
///     rewrite: a `tt.return`/`tt.call` operand (`FuncOp` signatures stay legal
///     here), an op outside the dialects in the conversion target (ttng/AMD
///     descriptor ops are implicitly legal under partial conversion), or a
///     value whose provenance we cannot fully name (function argument, call
///     result, unknown op). Each would leave a consumer expecting a descriptor
///     we deleted.
///   - Otherwise, if any member is loop-recreated, demote the whole class: the
///     merge is consumed in the loop, so it pays the per-iteration create.
static void growCandidatesToMergeClosure(Operation *root,
                                         MakeDescSet &candidates) {
  llvm::EquivalenceClasses<Operation *> classes;
  llvm::DenseSet<Operation *> escaping;
  for (triton::MakeTensorDescOp desc : candidates)
    classes.insert(desc);

  root->walk([&](Operation *op) {
    // Mirror of the conversion target below: only these dialects are subject to
    // rewriting, and `tt.return`/`tt.call` reach a signature we keep legal. An
    // unregistered op (null dialect) is likewise something we cannot rewrite.
    Dialect *dialect = op->getDialect();
    bool isBoundary =
        isa<triton::ReturnOp, triton::CallOp>(op) || !dialect ||
        !isa<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
             mlir::triton::TritonDialect>(dialect);
    for (Value operand : op->getOperands()) {
      if (!isa<triton::TensorDescType>(operand.getType()))
        continue;
      // Partial provenance still tells us which makes are entangled, so union
      // it — but an incomplete trace means an unnameable descriptor merges in
      // here too, which pins the whole group to TD.
      MakeDescSet makes;
      bool complete = collectMakeTensorDescOps(operand, makes);
      Operation *leader = nullptr;
      for (triton::MakeTensorDescOp make : makes) {
        classes.insert(make);
        if (leader)
          classes.unionSets(leader, make);
        else
          leader = make;
        if (isBoundary || !complete)
          escaping.insert(make);
      }
    }
  });

  llvm::DenseSet<Operation *> seeded(candidates.begin(), candidates.end());
  MakeDescSet closed;
  for (auto it = classes.begin(), e = classes.end(); it != e; ++it) {
    if (!(*it)->isLeader())
      continue;
    bool hasCandidate = false, escapes = false;
    SmallVector<triton::MakeTensorDescOp> members;
    for (auto mi = classes.member_begin(**it); mi != classes.member_end();
         ++mi) {
      members.push_back(cast<triton::MakeTensorDescOp>(*mi));
      hasCandidate |= seeded.contains(*mi);
      escapes |= escaping.contains(*mi);
    }
    if (!hasCandidate || escapes)
      continue; // nothing to demote, or must stay on TD
    for (triton::MakeTensorDescOp make : members)
      closed.insert(make);
  }
  candidates = std::move(closed);
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}

struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
  Value roundF32ToTF32;
};

Descriptor unpackDescriptor(TensorDescType type, ValueRange pack) {
  int rank = type.getShape().size();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 2 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by padding and TF32 rounding option values.");

  Descriptor res;
  res.base = pack[0];
  res.shape = pack.slice(1, rank);
  res.strides = pack.slice(1 + rank, rank);
  res.paddingOption = pack[1 + 2 * rank];
  res.roundF32ToTF32 = pack[2 + 2 * rank];
  return res;
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        triton::ExpandDimsOp::create(builder, loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI64Type());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, indexRowType, offset);
  Value range = triton::MakeRangeOp::create(builder, loc, indexI32RowType, 0,
                                            blockShape[dim]);
  Value i64Range = arith::ExtSIOp::create(builder, loc, indexRowType, range);

  Value offsets = arith::AddIOp::create(builder, loc, splatOffset, i64Range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = triton::SplatOp::create(
        builder, loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, offsets[i], splatStride);
    Value broadcasted = triton::BroadcastOp::create(
        builder, loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        triton::AddPtrOp::create(builder, loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

Value generateMaskFromOffsetRanges(OpBuilder &builder, const Location &loc,
                                   ArrayRef<std::int64_t> blockShape,
                                   Descriptor &desc, ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange = offsetRanges[i];

    // Compare with lower bound
    Value lowerBound = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getI64Type(), 0);
    Value splatLowerBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              offsetWithRange, splatLowerBound);

    // Compare with upper bound
    Value splatUpperBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), desc.shape[i]);
    Value cmpUpper =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                              offsetWithRange, splatUpperBound);

    // And and broadcast
    Value andResult = arith::AndIOp::create(builder, loc, cmpLower, cmpUpper);
    Value broadcasted =
        triton::BroadcastOp::create(builder, loc, maskTensorType, andResult);

    // And up all results
    if (!mask) {
      mask = broadcasted;
    } else {
      mask = arith::AndIOp::create(builder, loc, mask, broadcasted);
    }
  }

  return mask;
}

Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                   ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                      offsetRanges);
}

Value generateOther(OpBuilder &builder, Location loc, Type scalarTy,
                    ArrayRef<int64_t> blockShape,
                    Value paddingOption = nullptr) {
  auto blockTy = RankedTensorType::get(blockShape, scalarTy);
  if (paddingOption && mlir::isa<FloatType>(scalarTy)) {
    auto floatTy = mlir::cast<FloatType>(scalarTy);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getFloatAttr(floatTy, nan)));
    auto zeroValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getZeroAttr(floatTy)));
    return mlir::arith::SelectOp::create(builder, loc, paddingOption, nanValue,
                                         zeroValue);
  } else {
    auto attr = builder.getZeroAttr(blockTy);
    return arith::ConstantOp::create(builder, loc, attr);
  }
}

Value generateOther(OpBuilder &builder, Location loc, TensorDescType descTy,
                    Value paddingOption = nullptr) {
  auto blockTy = descTy.getSignlessBlockType();
  return generateOther(builder, loc, blockTy.getElementType(),
                       blockTy.getShape(), paddingOption);
}

Type getI32TypeLike(OpBuilder &builder, Type ty) {
  if (auto shapedTy = dyn_cast<ShapedType>(ty))
    return shapedTy.clone(builder.getI32Type());
  return builder.getI32Type();
}

Value getI32ConstLike(OpBuilder &builder, Location loc, Type likeType,
                      int32_t value) {
  auto i32Ty = getI32TypeLike(builder, likeType);
  if (auto shapedTy = dyn_cast<ShapedType>(i32Ty)) {
    auto attr =
        DenseElementsAttr::get(shapedTy, builder.getI32IntegerAttr(value));
    return arith::ConstantOp::create(builder, loc, shapedTy, attr);
  }
  return arith::ConstantOp::create(builder, loc, i32Ty,
                                   builder.getI32IntegerAttr(value));
}

Value roundF32ToTF32(OpBuilder &builder, Location loc, Value value) {
  auto valueTy = value.getType();
  auto i32Ty = getI32TypeLike(builder, valueTy);
  auto bits = triton::BitcastOp::create(builder, loc, i32Ty, value);

  auto expMask = getI32ConstLike(builder, loc, i32Ty, 0x7F800000);
  auto exp = arith::AndIOp::create(builder, loc, bits, expMask);
  auto isSpecial = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         exp, expMask);

  auto shift = getI32ConstLike(builder, loc, i32Ty, 13);
  auto lsb = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, bits, shift),
      getI32ConstLike(builder, loc, i32Ty, 1));
  auto roundBias = arith::AddIOp::create(
      builder, loc, lsb, getI32ConstLike(builder, loc, i32Ty, 0x00000FFF));
  auto rounded = arith::AndIOp::create(
      builder, loc, arith::AddIOp::create(builder, loc, bits, roundBias),
      getI32ConstLike(builder, loc, i32Ty, 0xFFFFE000));
  auto outBits =
      arith::SelectOp::create(builder, loc, isSpecial, bits, rounded);
  return triton::BitcastOp::create(builder, loc, valueTy, outBits);
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
}

struct RewriteMakeTensorDesc : OpConversionPattern<triton::MakeTensorDescOp> {
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStridesPaddingOption;
    llvm::append_values(ptrShapeStridesPaddingOption, adaptor.getBase());
    llvm::append_range(ptrShapeStridesPaddingOption,
                       castToI64(rewriter, adaptor.getShape()));
    llvm::append_range(ptrShapeStridesPaddingOption, adaptor.getStrides());
    auto paddingOption = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    llvm::append_values(ptrShapeStridesPaddingOption, paddingOption);
    auto roundF32ToTF32 = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(false));
    llvm::append_values(ptrShapeStridesPaddingOption, roundF32ToTF32);
    rewriter.replaceOpWithMultiple(op, {ptrShapeStridesPaddingOption});
    return mlir::success();
  }
};

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto other = generateOther(rewriter, loc, descTy, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, generatePtr(rewriter, loc, blockShape, desc, offsets),
        generateMask(rewriter, loc, blockShape, desc, offsets), other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getElementType().isF32()) {

      auto ifOp = scf::IfOp::create(rewriter, loc, result.getType(),
                                    desc.roundF32ToTF32, /*withElse=*/true);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      auto rounded = roundF32ToTF32(rewriter, loc, result);
      scf::YieldOp::create(rewriter, loc, rounded);

      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      scf::YieldOp::create(rewriter, loc, result);
      result = ifOp.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteStorePattern : OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::pair<Value, Value>
generateGatherScatterPtrMask(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> blockShape, Descriptor &desc,
                             Value xOffsets, Value yOffset) {
  Value xOffsetRange =
      expandOffsets(builder, loc, blockShape, xOffsets, /*dim=*/0);
  yOffset = castToI64(builder, {yOffset})[0];
  auto xOffsetI64Ty = RankedTensorType::get(
      cast<RankedTensorType>(xOffsetRange.getType()).getShape(),
      yOffset.getType());
  xOffsetRange =
      arith::ExtSIOp::create(builder, loc, xOffsetI64Ty, xOffsetRange);
  auto yOffsetRange =
      getExpandedOffsetWithRange(builder, loc, blockShape, yOffset, /*dim=*/1);
  auto ptr = generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                         {xOffsetRange, yOffsetRange});
  auto mask = generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                           {xOffsetRange, yOffsetRange});
  return {ptr, mask};
}

struct RewriteGatherPattern : OpConversionPattern<triton::DescriptorGatherOp> {
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getResult().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto other = generateOther(rewriter, loc,
                               descTy.getSignlessBlockType().getElementType(),
                               blockShape, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, ptr, mask, other, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getSignlessBlockType().getElementType().isF32()) {
      auto rounded = roundF32ToTF32(rewriter, loc, result);
      result = arith::SelectOp::create(rewriter, loc, desc.roundF32ToTF32,
                                       rounded, result);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteScatterPattern
    : OpConversionPattern<triton::DescriptorScatterOp> {
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getSrc().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, ptr, op.getSrc(), mask, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::optional<RMWOp> translateReduceKind(DescriptorReduceKind kind,
                                         TensorDescType ty) {
  auto scalarTy = ty.getElementType();
  switch (kind) {
  case DescriptorReduceKind::ADD:
    return scalarTy.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMIN;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MIN;
    }
    return {};
  case DescriptorReduceKind::MAX:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMAX;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MAX;
    }
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    break;
  }
  return {};
}

struct RewriteReducePattern : OpConversionPattern<triton::DescriptorReduceOp> {
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto rmwOp = translateReduceKind(op.getKind(), descTy);
    if (!rmwOp) {
      std::string msgstring;
      llvm::raw_string_ostream msg(msgstring);
      msg << "Cannot fallback on descriptor atomic op, unsupported for type "
          << descTy.getElementType();
      return op->emitError(msgstring);
    }

    triton::AtomicRMWOp::create(
        rewriter, loc, descTy.getSignlessBlockType(), *rmwOp,
        generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        MemSemantic::RELEASE, MemSyncScope::GPU);
    op.erase();
    return success();
  }
};

/**
 * @brief This implements the pass for converting triton tensor descriptor
 * loads/stores into indexed loads/stores.
 *
 * The key idea is that each tensor descriptor can be broken down into multiple
 * values. Suppose we have a tensor pointer with rank r, we can cast that tensor
 * descriptor value to and from 1+2r values: a tensor pointer value and two i32
 * value for each dimension representing the dynamic shape and strides.
 *
 * As in normal conversion patterns, individual operations can be converted
 * using casted tensor descriptors and offsets and casting the results back to
 * tensor pointers.
 *
 * We have special handling for TMA loads/stores and the make tensor descriptor
 * op.
 *
 * @note Why use the conversion pattern rewriter? In most cases the defining
 * operation of a tensor descriptor will be a make tensor descriptor op.
 * However, this isn't always true - for example, if the tensor descriptor is a
 * function argument or is in a conditional statement, we need better tracking
 * of the pointer, shape, and strides.
 */
class TritonRewriteTensorDescriptorToPointerPass
    : public impl::TritonRewriteTensorDescriptorToPointerBase<
          TritonRewriteTensorDescriptorToPointerPass> {
public:
  using TritonRewriteTensorDescriptorToPointerBase::
      TritonRewriteTensorDescriptorToPointerBase;

  void runOnOperation() override {
    auto op = getOperation();

    // loop-recreated-only mode: only these descriptors (and ops tracing to
    // them) are illegal; everything else keeps the TMA path.
    llvm::SmallSetVector<triton::MakeTensorDescOp, 4> candidates;
    if (loopRecreatedOnly) {
      op->walk([&](triton::MakeTensorDescOp desc) {
        if (isLoopRecreatedDescriptor(desc))
          candidates.insert(desc);
      });
      // No loop-recreated descriptor: leave the module untouched (all TMA).
      if (candidates.empty())
        return;

      // Widen the seeds so merged descriptors are demoted as a unit, or dropped
      // entirely when the merge group must stay on TD.
      growCandidatesToMergeClosure(op, candidates);
      if (candidates.empty())
        return;
    }

    // True if op's descriptor operands all trace (only) to candidates.
    auto usesCandidateDesc = [&](mlir::Operation *o) {
      for (Value operand : o->getOperands()) {
        if (!isa<triton::TensorDescType>(operand.getType()))
          continue;
        auto descs = findAllMakeTensorDescOps(operand);
        if (descs.empty())
          return false; // untraceable -> not a candidate we own
        if (!llvm::all_of(descs,
                          [&](auto d) { return candidates.contains(d); }))
          return false;
      }
      return true;
    };

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<
        mlir::arith::ArithDialect, mlir::scf::SCFDialect,
        mlir::triton::TritonDialect>([&](mlir::Operation *op) {
      if (!hasATensorDescriptorType(op->getOperandTypes()) &&
          !hasATensorDescriptorType(op->getResultTypes()))
        return true; // no descriptor involved -> always legal
      if (!loopRecreatedOnly)
        return false; // static mode: every descriptor op is illegal (rewrite
                      // all)
      // Dynamic mode: illegal only if it derives from a candidate.
      return TypeSwitch<mlir::Operation *, bool>(op)
          .Case<triton::MakeTensorDescOp>(
              [&](auto d) { return !candidates.contains(d); })
          .Default([&](mlir::Operation *o) { return !usesCandidateDesc(o); });
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp funcOp) {
      // Candidates are loop-local, so signatures stay legal in dynamic mode;
      // static mode rewrites any descriptor-typed signature.
      if (loopRecreatedOnly)
        return true;
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into an pointer, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    FuncArgRenamer renamer(".");
    renamer.addRenamer([](mlir::triton::TensorDescType type,
                          llvm::SmallVectorImpl<std::string> &out_suffix) {
      auto tensorType = type.getSignlessBlockType();
      int dims = tensorType.getRank();
      out_suffix.push_back("");
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("shape." + std::to_string(i));
      }
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("stride." + std::to_string(i));
      }
      out_suffix.push_back("padding");
      out_suffix.push_back("roundF32ToTF32");
      return success();
    });

    mlir::RewritePatternSet patterns(op->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, renamer, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns
        .add<RewriteMakeTensorDesc, RewriteLoadPattern, RewriteStorePattern,
             RewriteGatherPattern, RewriteScatterPattern, RewriteReducePattern>(
            converter, &getContext());

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            op, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton
