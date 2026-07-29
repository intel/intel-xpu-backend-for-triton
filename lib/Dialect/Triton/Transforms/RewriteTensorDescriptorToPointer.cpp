#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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

/// Provenance of one descriptor-typed value: which `MakeTensorDescOp`s can
/// reach it, and whether that list is the whole story.
struct DescProvenance {
  MakeDescSet makes;
  /// True once a descriptor we cannot name reaches this value (a function
  /// argument, a call result, an op we do not model). Top of the lattice: no
  /// set of makes describes the value, so it can never be safely rewritten.
  bool unknown = false;

  /// Merge `other` in. Returns true if this grew, i.e. users must be revisited.
  bool join(const DescProvenance &other) {
    bool changed = false;
    if (other.unknown && !unknown) {
      unknown = true;
      changed = true;
    }
    for (triton::MakeTensorDescOp make : other.makes)
      changed |= makes.insert(make);
    return changed;
  }
};

/// Collect into `tied` the values that a structured control-flow op ties to its
/// `idx`th result: region args, the result, and the yielded/init values.
/// Mirrors `mlir::triton::gpu::getTiedArgs` — TritonTransforms cannot link
/// against TritonGPUTransforms, so the SCF plumbing is repeated here.
///
/// Returns false when `op` is not one we model, or when its arity does not fit
/// this single-index shape (an `scf.while` whose before- and after-regions
/// carry different numbers of values). Callers must treat that as "we cannot
/// say where the value resurfaces" rather than "it resurfaces nowhere".
static bool getTiedDescValues(Operation *op, unsigned idx,
                              SmallVectorImpl<Value> &tied) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (idx >= forOp.getInitArgs().size())
      return false;
    tied.append({forOp.getRegionIterArg(idx), forOp.getResult(idx),
                 forOp.getBody()->getTerminator()->getOperand(idx),
                 forOp.getInitArgs()[idx]});
    return true;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    // One index only names a tied group when inits/before-args and
    // results/after-args have the same arity, which scf.while does not require.
    ValueRange condArgs = whileOp.getConditionOp().getArgs();
    if (whileOp.getBeforeArguments().size() != condArgs.size() ||
        idx >= condArgs.size())
      return false;
    tied.append({whileOp.getBeforeArguments()[idx],
                 whileOp.getAfterArguments()[idx], whileOp.getResult(idx),
                 condArgs[idx], whileOp.getOperand(idx)});
    return true;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (idx >= ifOp.getNumResults())
      return false;
    tied.push_back(ifOp.getResult(idx));
    for (Region *rgn : {&ifOp.getThenRegion(), &ifOp.getElseRegion()})
      for (Block &block : *rgn)
        if (isa<scf::YieldOp>(block.getTerminator()))
          tied.push_back(block.getTerminator()->getOperand(idx));
    return true;
  }
  return false;
}

/// Number of leading operands of a loop op that are not tied to its results
/// (`scf.for` takes lower/upper/step first; `scf.while` takes only inits).
static unsigned getLoopOperandOffset(Operation *op) {
  return isa<scf::ForOp>(op) ? 3 : 0;
}

/// Forward-propagate descriptor provenance from every `MakeTensorDescOp` in
/// `root` to every descriptor-typed value it can reach, to a fixed point.
///
/// Direction matters for more than cost. A backward "which makes does this
/// value come from" query, run per consumer, has to answer two questions at
/// once — the set of makes *and* whether the set is complete — and callers that
/// disagree about how to treat an incomplete answer silently disagree about
/// legality. Propagating forward computes both once per value: merges are set
/// unions at the join points, and an unnameable source is just `unknown`
/// riding along the same edges.
///
/// Descriptor values reachable from no make at all come back `unknown`, so
/// callers never have to distinguish "absent" from "unnameable".
static llvm::DenseMap<Value, DescProvenance>
computeDescProvenance(Operation *root) {
  llvm::DenseMap<Value, DescProvenance> provenance;
  llvm::SetVector<Value> worklist;

  // Update `val`'s provenance and re-queue it if it grew.
  auto update = [&](Value val, const DescProvenance &info) {
    if (!isa<triton::TensorDescType>(val.getType()))
      return;
    if (provenance[val].join(info))
      worklist.insert(val);
  };

  // Poison every descriptor `op` introduces. Results and region arguments are
  // all the SSA values an operation can define, so this is a complete cut: a
  // descriptor flowing into `op` cannot reappear anywhere we have not marked.
  auto markUnknown = [&](Operation *op) {
    DescProvenance top;
    top.unknown = true;
    for (Value result : op->getResults())
      update(result, top);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Value arg : block.getArguments())
          update(arg, top);
  };

  // Whether the propagation below is the thing that defines `val`'s provenance.
  // Everything else is poisoned up front, so a descriptor never *looks* fully
  // named just because the op that produced it is one we do not model — else a
  // merge of a make with, say, a `ub.poison` descriptor would report clean
  // provenance and get rewritten out from under the poison operand.
  auto isPropagated = [](Value val) {
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      // Only a loop's entry block carries descriptors we thread; a func entry
      // block, or a block introduced by unstructured control flow, does not.
      return arg.getOwner()->isEntryBlock() &&
             isa<scf::ForOp, scf::WhileOp>(arg.getOwner()->getParentOp());
    }
    return isa<scf::ForOp, scf::WhileOp, scf::IfOp, arith::SelectOp,
               UnrealizedConversionCastOp>(cast<OpResult>(val).getOwner());
  };

  // Seed: every make names itself; every descriptor we cannot name is unknown.
  root->walk([&](Operation *op) {
    if (auto make = dyn_cast<triton::MakeTensorDescOp>(op)) {
      DescProvenance self;
      self.makes.insert(make);
      update(make.getResult(), self);
      return;
    }
    DescProvenance top;
    top.unknown = true;
    for (Value result : op->getResults())
      if (!isPropagated(result))
        update(result, top);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Value arg : block.getArguments())
          if (!isPropagated(arg))
            update(arg, top);
  });

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    DescProvenance info = provenance[val];

    for (OpOperand &use : val.getUses()) {
      Operation *owner = use.getOwner();
      Operation *parent = nullptr;
      unsigned idx = 0;
      if (isa<scf::ForOp, scf::WhileOp>(owner)) {
        // Into a loop: an init operand feeds the region arg, result, and yield.
        parent = owner;
        idx = use.getOperandNumber() - getLoopOperandOffset(owner);
      } else if (isa<scf::YieldOp>(owner)) {
        // Out of a region: a yielded value feeds the parent's result and, for
        // loops, the next iteration's region arg.
        parent = owner->getParentOp();
        idx = use.getOperandNumber();
      } else if (isa<scf::ConditionOp>(owner)) {
        parent = owner->getParentOp();
        idx = use.getOperandNumber() - 1; // operand 0 is the condition itself
      } else if (isa<arith::SelectOp, UnrealizedConversionCastOp>(owner)) {
        // A select/cast forwards its descriptor arms to its result.
        for (Value result : owner->getResults())
          update(result, info);
        continue;
      } else {
        // A terminal consumer (descriptor_load/store, return, call, a ttng op)
        // forwards nothing; any descriptor it does define was poisoned above.
        continue;
      }

      SmallVector<Value> tied;
      if (getTiedDescValues(parent, idx, tied))
        for (Value v : tied)
          update(v, info);
      else
        markUnknown(parent);
    }
  }
  return provenance;
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

/// Whether a group of entangled makes wants to be rewritten, and whether it is
/// allowed to be. Monotone: both bits only ever turn on.
struct DescDecision {
  bool demote = false; // some member is loop-recreated, so worth rewriting
  bool pin = false;    // some member has to stay a `!tt.tensordesc`

  /// Merge `other` in. Returns true if this changed.
  bool join(const DescDecision &other) {
    bool changed = (other.demote && !demote) || (other.pin && !pin);
    demote |= other.demote;
    pin |= other.pin;
    return changed;
  }
};

/// True if the conversion below can actually rewrite `op` when a descriptor it
/// touches is demoted: either a pattern matches it, or it is structural and the
/// type converter expands its signature.
///
/// An allowlist on purpose. The complement — `tt.return`/`tt.call` (`FuncOp`
/// signatures stay legal in this mode), ops outside the conversion target's
/// dialects (ttng/AMD descriptor ops, implicitly legal under partial
/// conversion), unregistered ops — is open-ended, and a new descriptor consumer
/// added to TritonDialect should default to pinning rather than to silently
/// keeping an operand we deleted.
static bool canRewriteWithDemotedDesc(Operation *op) {
  return isa<triton::MakeTensorDescOp, triton::DescriptorLoadOp,
             triton::DescriptorStoreOp, triton::DescriptorGatherOp,
             triton::DescriptorScatterOp, triton::DescriptorReduceOp,
             scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp,
             scf::ConditionOp, arith::SelectOp, UnrealizedConversionCastOp>(op);
}

/// Pick the `MakeTensorDescOp`s that loop-recreated-only mode rewrites.
///
/// Demotion is all-or-nothing per *entangled group*, because the unit of
/// rewriting is an operation, not a value. Two descriptors are entangled when
/// some op touches both: the type converter expands every `TensorDescType` in a
/// signature, so a loop carrying one demoted and one kept descriptor cannot be
/// half-converted. Merges (scf.if / select / iter-arg) are the same rule seen
/// from the other side — the merged value appears on an op, so its whole
/// provenance is entangled. Getting this wrong leaves a consumer expecting a
/// `!tt.tensordesc` that no longer exists, and the type converter runs with
/// `buildMaterializations = false`, so nothing bridges the gap.
///
/// So group by op, then propagate two facts over the groups to a fixed point:
///   - `demote`, seeded at the loop-recreated makes.
///   - `pin`, seeded wherever the group cannot be rewritten: an op that is not
///     `canRewriteWithDemotedDesc`, or a descriptor whose provenance we cannot
///     fully name (function argument, call result, an op we do not model).
/// A make is rewritten iff it comes out `demote` and not `pin`. Iterating to a
/// fixed point is what makes grouping transitive: groups can overlap without
/// being equal, so `{A,B}` and `{B,C}` decide together.
static MakeDescSet
computeDemotions(Operation *root,
                 const llvm::DenseMap<Value, DescProvenance> &provenance) {
  llvm::DenseMap<Operation *, DescDecision> decisions;
  SmallVector<MakeDescSet> groups;

  // The descriptors worth rewriting: rebuilt every iteration of some loop.
  root->walk([&](triton::MakeTensorDescOp make) {
    if (isLoopRecreatedDescriptor(make))
      decisions[make].demote = true;
  });

  root->walk([&](Operation *op) {
    // Every descriptor this op touches decides with the others.
    MakeDescSet group;
    bool pin = !canRewriteWithDemotedDesc(op);
    auto add = [&](Value val) {
      if (!isa<triton::TensorDescType>(val.getType()))
        return;
      auto it = provenance.find(val);
      // Unnameable provenance pins whatever of it we *could* name.
      if (it == provenance.end() || it->second.unknown)
        pin = true;
      if (it != provenance.end())
        group.insert(it->second.makes.begin(), it->second.makes.end());
    };
    for (Value operand : op->getOperands())
      add(operand);
    for (Value result : op->getResults())
      add(result);
    if (group.empty())
      return;
    if (pin)
      for (triton::MakeTensorDescOp make : group)
        decisions[make].pin = true;
    if (group.size() > 1)
      groups.push_back(std::move(group)); // a lone make constrains only itself
  });

  // Unify both bits within each group. Sweeps are bounded by the longest chain
  // of overlapping groups, which is 1 for the shapes that occur in practice.
  for (bool changed = true; changed;) {
    changed = false;
    for (const MakeDescSet &group : groups) {
      DescDecision merged;
      for (triton::MakeTensorDescOp make : group)
        merged.join(decisions[make]);
      for (triton::MakeTensorDescOp make : group)
        changed |= decisions[make].join(merged);
    }
  }

  // Walk again rather than iterating `decisions`, whose order is not stable.
  MakeDescSet demotions;
  root->walk([&](triton::MakeTensorDescOp make) {
    auto it = decisions.find(make);
    if (it != decisions.end() && it->second.demote && !it->second.pin)
      demotions.insert(make);
  });
  return demotions;
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

    // loop-recreated-only mode: only these descriptors (and the ops handling
    // them) are illegal; everything else keeps the TMA path. One forward
    // provenance walk decides them all, so legality below is a lookup.
    MakeDescSet demotions;
    llvm::DenseMap<Value, DescProvenance> provenance;
    if (loopRecreatedOnly) {
      provenance = computeDescProvenance(op);
      demotions = computeDemotions(op, provenance);
      // Nothing to demote: leave the module untouched (all TMA).
      if (demotions.empty())
        return;
    }

    // True if every descriptor `o` touches is one we are rewriting. Grouping
    // above guarantees this is all-or-nothing per op, so any descriptor of `o`
    // being demoted implies all of them are.
    auto handlesDemotedDesc = [&](mlir::Operation *o) {
      auto isDemoted = [&](Value v) {
        if (!isa<triton::TensorDescType>(v.getType()))
          return true;
        auto it = provenance.find(v);
        if (it == provenance.end() || it->second.unknown)
          return false; // unnameable -> never ours
        return llvm::all_of(it->second.makes,
                            [&](auto d) { return demotions.contains(d); });
      };
      return llvm::all_of(o->getOperands(), isDemoted) &&
             llvm::all_of(o->getResults(), isDemoted);
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
      // Dynamic mode: illegal only if it handles a descriptor we demote.
      return TypeSwitch<mlir::Operation *, bool>(op)
          .Case<triton::MakeTensorDescOp>(
              [&](auto d) { return !demotions.contains(d); })
          .Default([&](mlir::Operation *o) { return !handlesDemotedDesc(o); });
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp funcOp) {
      // Signatures stay legal in dynamic mode — that is what pins any group
      // reaching one — while static mode rewrites descriptor-typed signatures.
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
