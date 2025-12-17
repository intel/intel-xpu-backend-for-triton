#include "third_party/intel/include/Analysis/Range.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>

#define DEBUG_TYPE "triton-intel-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {

constexpr int64_t kDefaultMaxTripCount = 1024;
constexpr uint64_t kDefaultMaxPrograms = INT_MAX;

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, tt::GetProgramIdOp, tt::GetNumProgramsOp>::value>>
static void inferResultRange(OpType op, uint64_t max,
                             SetIntRangeFn setResultRange) {
  Value result = op.getResult();
  IntegerType resTy = cast<IntegerType>(result.getType());
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(resTy);
  setResultRange(
      result, ConstantIntRanges::range({bitWidth, 0, resTy.isSigned()},   // min
                                       {bitWidth, max, resTy.isSigned()}, // max
                                       resTy.isSigned()));
}

static void inferResultRange(tt::MakeRangeOp op, SetIntRangeFn setResultRange) {
  TypedValue<RankedTensorType> result = op.getResult();
  RankedTensorType resTy = result.getType();
  assert(isa<IntegerType>(resTy.getElementType()) && "expected int type");

  IntegerType elTy = cast<IntegerType>(resTy.getElementType());
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(elTy);
  setResultRange(result,
                 ConstantIntRanges::range(
                     {bitWidth, op.getStart(), elTy.isSigned()},   // min
                     {bitWidth, op.getEnd() - 1, elTy.isSigned()}, // max
                     elTy.isSigned()));
}

static void inferResultRange(tt::GatherOp op,
                             ArrayRef<ConstantIntRanges> argRanges,
                             SetIntRangeFn setResultRange) {
  assert(argRanges.size() == 2 && "expected two arg ranges");
  setResultRange(op.getResult(), argRanges[0]);
}

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, tt::TransOp, tt::SplitOp, tt::BroadcastOp,
              tt::ExpandDimsOp, tt::SplatOp, tt::ReshapeOp,
              ttg::ConvertLayoutOp, tt::JoinOp, tt::CatOp>::value>>
static void inferResultRange(OpType op, ArrayRef<ConstantIntRanges> argRanges,
                             SetIntRangeFn setResultRange) {
  if constexpr (llvm::is_one_of<OpType, tt::JoinOp, tt::CatOp>::value) {
    assert(op.getNumOperands() == 2 && argRanges.size() == 2 &&
           "expecting two operands");
    for (Value result : op->getResults())
      setResultRange(result, argRanges[0].rangeUnion(argRanges[1]));
  } else {
    for (Value result : op->getResults())
      setResultRange(result, argRanges[0]);
  }
}

static void inferResultRange(HistogramOp op, SetIntRangeFn setResultRange) {
  for (Value result : op->getResults()) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(result.getType());
    setResultRange(result, ConstantIntRanges::fromSigned(
                               APInt::getZero(bitWidth).sext(bitWidth),
                               APInt::getMaxValue(bitWidth).sext(bitWidth)));
  }
}

static DenseMap<Value, SetVector<Operation *>>
collectAssumptions(Operation *rootOp, bool filterConstants = true) {
  DenseMap<Value, SetVector<Operation *>> assumptions;
  rootOp->walk([&](LLVM::AssumeOp op) {
    Operation *defOp = op.getCond().getDefiningOp();
    for (auto operand : defOp->getOperands()) {
      if (filterConstants && getConstantIntValue(operand))
        continue;
      assumptions[operand].insert(defOp);
    }
  });
  return assumptions;
}

static std::optional<ConstantIntRanges>
getAssumedRange(const SetVector<Operation *> &assumptions, Value val,
                Block *useBlock, const DominanceInfo &domInfo) {
  std::optional<ConstantIntRanges> result;
  for (Operation *assumption : assumptions) {
    arith::CmpIOp cmpOp = dyn_cast<arith::CmpIOp>(assumption);
    if (!cmpOp) {
      emitRemark(assumption->getLoc(), "unsupported operation");
      continue;
    }
    if (!useBlock || !domInfo.dominates(cmpOp->getBlock(), useBlock))
      continue;

    if (auto assumedRange = tt::getBoundFromCmpOp(cmpOp, val)) {
      if (result)
        result = (*result).intersection(*assumedRange);
      else
        result = *assumedRange;
    }
  }

  if (result) {
    ConstantIntRanges &range = *result;
    if (range.smin().isNonNegative()) {
      // Consider 0 <= x <= 1024.
      // When processing x > 0, the value range of x is
      //  vr1={umin=0, umax=0xf...f, smin=0, smax=0x7...f}
      // When processing x < 1024, the value range of x is:
      //  vr2={umin=0, umax=0xf...f, smin=..., smax=1024}
      // and
      //  vr1 ∩ vr2 = {umin=0, umax=0xf...f, smin=0, smax=1024}
      // note that the umax=0xf...f is annoying, need to change to 1024.
      return ConstantIntRanges::range(range.smin(), range.smax(), true);
    }
  }
  return result;
}

// Check if any of the bounds have zero bit width, indicating an empty range.
static bool isEmpty(ConstantIntRanges range) {
  return range.umin().getBitWidth() == 0 || range.umax().getBitWidth() == 0 ||
         range.smin().getBitWidth() == 0 || range.smax().getBitWidth() == 0;
}

/// Construct the narrowest range of \p val using assumptions the given
/// value participates in.
/// For example, given:
///   %is_sge = arith.cmpi sge, %val, %c0 : i32
///   llvm.intr.assume %is_sge : i1
///   %is_sle = arith.cmpi sle, %val, %c128 : i32
///   llvm.intr.assume %is_slt : i1
/// the range of `val` is:
///   [0, INT_MAX] ∩ [INT_MIN, 128] = [0, 128]
static std::optional<ConstantIntRanges> getAssumedRange(
    Value val, Block *useBlock,
    const llvm::DenseMap<Value, SetVector<Operation *>> &assumptions,
    const DominanceInfo &domInfo) {
  if (!assumptions.contains(val))
    return std::nullopt;
  return getAssumedRange(assumptions.lookup(val), val, useBlock, domInfo);
}

IntegerRangeAnalysis::IntegerRangeAnalysis(Operation *top,
                                           DataFlowSolver &solver,
                                           DominanceInfo &domInfo)
    : dataflow::IntegerRangeAnalysis(solver), integerValues(), assumptions(),
      domInfo(domInfo) {
  assumptions = collectAssumptions(top);
}

void IntegerRangeAnalysis::setToEntryState(
    dataflow::IntegerValueRangeLattice *lattice) {
  Value anchor = lattice->getAnchor();
  Type elemType = getElementTypeOrSelf(anchor);
  if (!isa<IntegerType, IndexType>(elemType))
    return;

  auto getParentFunction = [](Value val) -> std::optional<tt::FuncOp> {
    Operation *definingOp = val.getDefiningOp();
    if (!definingOp)
      if (Block *block = val.getParentBlock())
        definingOp = block->getParentOp();

    if (definingOp) {
      if (auto funcOp = dyn_cast<tt::FuncOp>(definingOp))
        return funcOp;
      return definingOp->getParentOfType<tt::FuncOp>();
    }

    return std::nullopt;
  };

  std::optional<tt::FuncOp> funcOp = getParentFunction(anchor);
  assert(funcOp && "Could not find parent function ");

  IntegerValueRange range = IntegerValueRange::getMaxRange(anchor);
  Block *entryBlock = &funcOp->getBody().front();

  if (std::optional<ConstantIntRanges> assumedRange =
          getAssumedRange(anchor, entryBlock, assumptions, domInfo))
    range = *assumedRange;

  ChangeResult changed = lattice->join(range);

  LLVM_DEBUG({
    if (changed == ChangeResult::Change) {
      DBGS() << "Set range of ";
      anchor.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << " to " << range << "\n";
    }
  });

  propagateIfChanged(lattice, changed);
}

LogicalResult IntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  Block *block = op->getBlock();

  // Figure out the implied range of result and source operands.
  opResultAssumption.clear();
  for (OpResult result : op->getResults()) {
    if (std::optional<ConstantIntRanges> assumedRange =
            getAssumedRange(result, block, assumptions, domInfo))
      opResultAssumption.insert(std::pair(result, *assumedRange));
  }

  SmallVector<const dataflow::IntegerValueRangeLattice *, 4> opndRanges;
  SmallVector<std::unique_ptr<dataflow::IntegerValueRangeLattice>, 4>
      newSrcLattices;

  for (auto [index, opnd] : llvm::enumerate(op->getOperands())) {
    std::optional<ConstantIntRanges> assumedRange =
        getAssumedRange(opnd, block, assumptions, domInfo);
    if (!assumedRange) {
      opndRanges.push_back(operands[index]);
      continue;
    }

    auto newLattice =
        std::make_unique<dataflow::IntegerValueRangeLattice>(opnd);
    (void)newLattice->join(IntegerValueRange(*assumedRange));
    opndRanges.push_back(newLattice.get());
    newSrcLattices.push_back(std::move(newLattice));
  }
  assert(opndRanges.size() == operands.size() && "size disagree");

  // Infer the range and, if an assumed range is available, intersect the
  // assumed range with the inferred range.
  LogicalResult visitResult =
      visitOperationHelper(op, opndRanges, resultsLattices);

  // If previous the step failed to infer the range, apply the assumed range if
  // present.
  for (auto [index, lattice] : llvm::enumerate(resultsLattices)) {
    Value result = op->getResult(index);
    const auto assumedIter = opResultAssumption.find(result);
    if (assumedIter == opResultAssumption.end())
      continue;

    const IntegerValueRange &range = lattice->getValue();
    if (!range.isUninitialized() && !isEmpty(range.getValue()))
      continue;

    const ConstantIntRanges &assumedRange = assumedIter->second;
    IntegerValueRange newRange(assumedRange);
    ChangeResult changed = lattice->join(newRange);

    LLVM_DEBUG({
      if (changed == ChangeResult::Change) {
        DBGS() << ">Force apply assumed value range. value:";
        result.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << ", range:" << range << "\n";
      }
    });

    propagateIfChanged(lattice, changed);
  }

  return visitResult;
}

std::optional<int64_t>
IntegerRangeAnalysis::getTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  std::optional<Value> iv = loop.getSingleInductionVar();
  if (!iv)
    return std::nullopt;

  unsigned int width = ConstantIntRanges::getStorageBitwidth(iv->getType());

  auto getLoopRangeInfo = [&](std::optional<OpFoldResult> loopBound,
                              Block *block,
                              std::optional<bool> getUpper = std::nullopt,
                              std::optional<APInt> defaultVal = std::nullopt) {
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = dyn_cast_if_present<Value>(*loopBound)) {
        const dataflow::IntegerValueRangeLattice *lattice =
            getLatticeElementFor(getProgramPointBefore(block), value);
        if (lattice && !lattice->getValue().isUninitialized())
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }

    if (defaultVal)
      return *defaultVal;

    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  Block *block = iv->getParentBlock();
  APInt min = getLoopRangeInfo(lowerBound, block,
                               /*getUpper=*/false);
  APInt max = getLoopRangeInfo(upperBound, block,
                               /*getUpper=*/true);
  // We can assume step is 1 if no range information as that gives us the upper
  // bound of the number of iterations.
  APInt stepValDefault = {width, 1, /*isSigned=*/true};
  APInt stepVal =
      getLoopRangeInfo(step, block, /*getUpper=*/{}, stepValDefault);

  if (stepVal.isNegative())
    std::swap(min, max);
  // This is necessary to catch a case like this:
  //  # range = [0 1024]
  //  K = ....
  //  # range = [1, 64]
  //  k = ...
  //  # range = [0, 16] -> stepVal = range.smin() = 0
  //  step = ceildiv(K, k)
  if (stepVal.isZero())
    stepVal = stepValDefault;
  if (max.sge(min))
    return llvm::divideCeilSigned(max.getSExtValue() - min.getSExtValue(),
                                  stepVal.getSExtValue());
  return std::nullopt;
}

LogicalResult IntegerRangeAnalysis::visitOperationHelper(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  LDBG("Inferring ranges for " << *op);

  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &operands, &resultsLattices,
                       this](Value resultVal,
                             const IntegerValueRange &incomingRange) {
    // Preparation:
    //  - Get the lattice associated with given particular result value.
    //  - Make a copy of value-range just inferred, as we need to do some
    //   change to it before it's joined to the existing lattice.
    auto result = dyn_cast<OpResult>(resultVal);
    if (!result)
      return;

    assert(llvm::is_contained(op->getResults(), result));

    dataflow::IntegerValueRangeLattice *lattice =
        resultsLattices[result.getResultNumber()];
    IntegerValueRange newRange = incomingRange;

    // If there is assumed range, the assumed one take precedence.
    // TODO: I think this is bit conservative, the better way is:
    //  final_range = (old_range ∪ incomingRange) ∩ assume_range
    auto iter = opResultAssumption.find(resultVal);
    if (iter != opResultAssumption.end()) {
      const ConstantIntRanges &range = iter->second;
      if (std::optional<ConstantIntRanges> assumedRange =
              getAssumedRange(resultVal, op->getBlock(), assumptions, domInfo))
        newRange = IntegerValueRange(newRange.getValue().intersection(range));
    }

    // Update the range. Note that we are using `join` operation which means
    // `union`. Transfer function must be monotone! The resolver would otherwise
    // fall into an infinite loop.
    ChangeResult changed = lattice->join(newRange);

    LLVM_DEBUG({
      OpPrintingFlags flags;
      flags.skipRegions(true);
      DBGS() << ((changed == ChangeResult::Change) ? ">Inferred range for: "
                                                   : ">Remain unchanged: ");
      resultVal.printAsOperand(llvm::dbgs(), flags);
      llvm::dbgs() << ", resulting state:" << lattice->getValue()
                   << ", in value-range: " << newRange << "\n";
    });

    // step 4: Add those ops that depends on this op to the worklist. The
    // resolver will iterate all items in the worklist until it become empty.
    propagateIfChanged(lattice, changed);
  };

  // Ops with fixed ranges.
  if (isa<tt::GetProgramIdOp, tt::GetNumProgramsOp, tt::MakeRangeOp,
          tt::HistogramOp>(op)) {
    TypeSwitch<Operation *>(op)
        .Case<tt::GetProgramIdOp>([&](auto getProgramIdOp) {
          inferResultRange(getProgramIdOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<tt::GetNumProgramsOp>([&](auto getNumProgramsOp) {
          inferResultRange(getNumProgramsOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<tt::MakeRangeOp>([&](auto makeRangeOp) {
          inferResultRange(makeRangeOp, joinCallback);
        })
        .Case<tt::HistogramOp>([&](auto histogramOp) {
          inferResultRange(histogramOp, joinCallback);
        });
    return success();
  }

  SmallVector<IntegerValueRange> argIntValueRanges = map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  // Ops with input/output ranges.
  if (isa<tt::TransOp, tt::SplitOp, tt::BroadcastOp, tt::ExpandDimsOp,
          tt::SplatOp, tt::ReshapeOp, ttg::ConvertLayoutOp, tt::JoinOp,
          tt::CatOp, tt::GatherOp>(op)) {
    SmallVector<ConstantIntRanges> argConstIntRanges;
    for (const auto &r : argIntValueRanges) {
      if (r.isUninitialized()) {
        setAllToEntryStates(resultsLattices);
        return success();
      }
      argConstIntRanges.push_back(r.getValue());
    }

    llvm::TypeSwitch<Operation *>(op)
        .Case<tt::TransOp, tt::SplitOp, tt::BroadcastOp, tt::ExpandDimsOp,
              tt::SplatOp, tt::ReshapeOp, ttg::ConvertLayoutOp>([&](auto op) {
          inferResultRange(op, argConstIntRanges, joinCallback);
        })
        .Case<tt::JoinOp, tt::CatOp>([&](auto op) {
          inferResultRange(op, argConstIntRanges, joinCallback);
        })
        .Case<tt::GatherOp>([&](auto op) {
          inferResultRange(op, argConstIntRanges, joinCallback);
        });
    return success();
  }

  // TODO: It looks like inferResultRangesFromOptional does not handle several
  // operations well:
  //   - arith.shrui, e.g. arith.shrui %arg3, %c5_i32
  //
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    inferrable.inferResultRangesFromOptional(argIntValueRanges, joinCallback);
    return success();
  }

  setAllToEntryStates(resultsLattices);
  return success();
}

void IntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionSuccessor successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  LLVM_DEBUG({
    DBGS() << "Visit Region Succesors of ";
    OpPrintingFlags flags;
    flags.skipRegions(true);
    branch.print(llvm::dbgs(), flags);
    llvm::dbgs() << "\n";
  });

  SmallVector<dataflow::IntegerValueRangeLattice *> lattices;
  for (dataflow::AbstractSparseLattice *lattice : abstractLattices)
    lattices.push_back(
        static_cast<dataflow::IntegerValueRangeLattice *>(lattice));

  auto getTotalLoopTripCount = [this](LoopLikeOpInterface loop) {
    SmallVector<LoopLikeOpInterface> loops{loop};
    {
      Operation *parentOp = loop->getParentOp();
      while (parentOp) {
        if (isa<LoopLikeOpInterface>(parentOp))
          loops.push_back(cast<LoopLikeOpInterface>(parentOp));
        parentOp = parentOp->getParentOp();
      }
    }

    return std::accumulate(loops.begin(), loops.end(), 1ll,
                           [this](int64_t accum, LoopLikeOpInterface loop) {
                             return accum * getTripCount(loop).value_or(
                                                kDefaultMaxTripCount + 1);
                           });
  };

  // Initialize loop trip counts.
  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopTripCounts;
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;

  auto loop = dyn_cast<LoopLikeOpInterface>(branch.getOperation());
  if (loop) {
    if (!loopTripCounts.contains(loop)) {
      loopTripCounts[loop] = std::numeric_limits<int64_t>::max();
      for (auto lattice : lattices)
        loopVisits[{loop, lattice}] = 0ll;
    }

    int64_t loopTripCount = getTotalLoopTripCount(loop);

    LLVM_DEBUG({
      DBGS() << "Trip count for ";
      OpPrintingFlags flags;
      flags.skipRegions(true);
      loop->print(llvm::dbgs(), flags);
      llvm::dbgs() << "\n";
      DBGS() << " --> " << loopTripCount << '\n';
    });

    if (loopTripCount < loopTripCounts[loop])
      loopTripCounts[loop] = loopTripCount;
  }

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  // Note: It does not seems to be quite obvious; this loop could update SCF
  // operations' LHS. e.g. If the given "branch" argument is scf.if, and the
  // scf.if construct looks like following:
  //   x = scf.if cond
  //    m = ... // op_m
  //    yield m
  //   else
  //    n = ... // op_n
  //    yield n
  //
  // This loop tries to update lattice(x) = join(lattice(m), lattice(n),
  // provided lattice(m) and lattice(n) are initialized.
  //
  // Note that the state of lattice(m) and lattice(n) was updated in the
  // "previous" round. In this "round", the scf.if is visitied right now, and
  // it takes this moment to update its LHS.
  //
  // Alternatively, when we visit, say op_m, we notice its result is used by
  // a yieldOp, get the yieldOp's corresponding receiver, in this case x, and
  // update its state accordingly.
  //
  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    if (op == branch)
      operands = branch.getEntrySuccessorOperands(successor);
    else if (auto regionTerminator =
                 dyn_cast<RegionBranchTerminatorOpInterface>(op))
      operands = regionTerminator.getSuccessorOperands(successor);

    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0u;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty())
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();

        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(
                branch, branch->getResults().slice(firstIndex, inputs.size())),
            lattices, firstIndex);
      } else {
        if (!inputs.empty())
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();

        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #tripcount times, stop propagating.
      if (loop && loopVisits[loopArgLat] >= loopTripCounts[loop])
        continue;

      ChangeResult changed;
      if (loop && loopTripCounts[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).
        changed = argLat->join(IntegerValueRange::getMaxRange(oper));
      } else {
        // Else, propagate pred operands.
        auto operLat = *getLatticeElementFor(point, oper);
        changed = argLat->join(operLat);
        LLVM_DEBUG({
          if (changed == ChangeResult::Change) {
            DBGS() << "Operand lattice ";
            oper.printAsOperand(llvm::dbgs(), {});
            llvm::dbgs() << " --> " << operLat.getValue() << "\n";
          }
        });
      }

      propagateIfChanged(argLat, changed);

      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (loop && changed == ChangeResult::Change)
        ++loopVisits[loopArgLat];
    }
  }
}

void IntegerRangeAnalysis::initializeModule(ModuleOp &mod) {
  mod.walk<WalkOrder::PreOrder>([&](FuncOp funcOp) {
    Block *entryBlock = &funcOp.getBody().front();
    for (BlockArgument argument : funcOp.getArguments()) {
      if (!assumptions.count(argument))
        continue;

      dataflow::IntegerValueRangeLattice *argLattice =
          getLatticeElement(argument);

      IntegerValueRange range = IntegerValueRange::getMaxRange(argument);
      if (auto assumedRange =
              getAssumedRange(argument, entryBlock, assumptions, domInfo))
        range = *assumedRange;

      // The lattice must in "bottom" state, The join() operation is to set
      // the state to the given "range".
      assert(argLattice->getValue().isUninitialized() &&
             "lattice must be in bottom state");
      (void)argLattice->join(range);
    }
  });
}

} // namespace mlir::triton::intel
