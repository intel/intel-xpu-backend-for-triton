#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUTAILUNALIGNEDWIDTHLOADS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Spread the rows over the lanes, else each lane loads every row.
static Attribute getLaneDistributedEncoding(RankedTensorType sliceTy) {
  MLIRContext *ctx = sliceTy.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(sliceTy);
  StringAttr kLane = StringAttr::get(ctx, "lane");
  tt::LinearLayout::BasesT bases = ll.getBases();
  auto &regs = bases[StringAttr::get(ctx, "register")];
  auto &lanes = bases[kLane];
  if (regs.size() < lanes.size() ||
      !ll.sublayoutIsZero({kLane}, llvm::to_vector(ll.getOutDimNames())))
    return {};
  std::copy_n(regs.begin(), lanes.size(), lanes.begin());
  regs.erase(regs.begin(), regs.begin() + lanes.size());
  std::optional<tt::LinearLayout> distributed = tt::LinearLayout::tryCreate(
      std::move(bases), ll.getOutDims(), /*requireSurjective=*/true);
  if (!distributed)
    return {};
  Attribute enc = ttg::LinearEncodingAttr::get(ctx, std::move(*distributed));
  if (!ttgi::cvtIsSubGroupShuffle(sliceTy.cloneWithEncoding(enc), sliceTy))
    return {};
  return enc;
}

// An attribute, so that loads cloned by the loop split inherit it.
enum class Mode : int32_t { Guarded, Unguarded, None };
constexpr StringLiteral kModeAttr = "ttig.tail_patch_mode";

Mode getMode(Operation *op) {
  return Mode(cast<IntegerAttr>(op->getAttr(kModeAttr)).getInt());
}

void setMode(Operation *op, Mode mode) {
  op->setAttr(kModeAttr,
              Builder(op->getContext()).getI32IntegerAttr(int32_t(mode)));
}

struct UnalignedWidthLoad {
  tt::DescriptorLoadOp load;
  tt::MakeTensorDescOp make;
  int64_t width, granule, prefix;

  static std::optional<UnalignedWidthLoad> match(tt::DescriptorLoadOp load) {
    auto make = load.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
    RankedTensorType tileTy = load.getType();
    Type elemTy = tileTy.getElementType();
    // Packed fp4 has no bit width.
    if (!make || tileTy.getRank() != 2 || make.getShape().size() != 2 ||
        load.getIndices().size() != 2 ||
        load.getDesc().getType().getSignlessBlockType().getShape() !=
            tileTy.getShape() ||
        !isa_and_nonnull<ttg::DistributedEncodingTrait>(tileTy.getEncoding()) ||
        make.getPadding() != tt::PaddingOption::PAD_ZERO ||
        !elemTy.isIntOrFloat() || elemTy.getIntOrFloatBitWidth() % 8 ||
        make.getBase().getType().getPointeeType() != elemTy ||
        !tt::intel::isConstant(make.getStrides()[1], 1))
      return std::nullopt;
    std::optional<int64_t> width =
        tt::intel::getFoldedConstantValue(make.getShape()[1]);
    if (!width)
      return std::nullopt;
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    int64_t granule =
        ttgi::get2DBlockIOWidthGranuleBytes(elemBytes * 8) / elemBytes;
    int64_t prefix = llvm::alignDown(*width, granule);
    // A block surface is at least 64 bytes.
    if (prefix == *width || prefix * elemBytes < 64)
      return std::nullopt;
    return UnalignedWidthLoad{load, make, *width, granule, prefix};
  }
};

// SymbolicBoundsProver misses `off += BLOCK_K` in a unit-step loop.
struct LoopSplit {
  scf::ForOp forOp;
  int64_t step;
  SmallVector<std::pair<UnalignedWidthLoad, int64_t>> entries; // load, slope
  DenseMap<Value, std::optional<int64_t>> slopes;
  IRMapping first;

  LoopSplit(scf::ForOp forOp, int64_t step) : forOp(forOp), step(step) {}

  std::optional<int64_t> slope(Value v) {
    if (forOp.isDefinedOutsideOfLoop(v))
      return 0;
    if (auto it = slopes.find(v); it != slopes.end())
      return it->second;
    return slopes[v] = computeSlope(v);
  }

  Value firstIteration(OpBuilder &b, Value v) {
    if (Value f = first.lookupOrNull(v); f || forOp.isDefinedOutsideOfLoop(v))
      return f ? f : v;
    Operation *op = v.getDefiningOp();
    for (Value operand : op->getOperands())
      firstIteration(b, operand);
    b.clone(*op, first);
    return first.lookup(v);
  }

private:
  std::optional<int64_t> computeSlope(Value v) {
    if (v == forOp.getInductionVar())
      return step;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Value inc =
          arg.getOwner() == forOp.getBody() && arg.getType().isIntOrIndex()
              ? tt::intel::getIterArgIncrement(forOp, arg)
              : Value();
      return inc ? tt::intel::getFoldedConstantValue(inc) : std::nullopt;
    }
    // Speculated before the loop, so pure.
    Operation *op = v.getDefiningOp();
    if (op->getNumRegions() || !isPure(op))
      return std::nullopt;
    SmallVector<int64_t> s;
    for (Value operand : op->getOperands()) {
      std::optional<int64_t> sv = slope(operand);
      if (!sv)
        return std::nullopt;
      s.push_back(*sv);
    }
    if (llvm::all_of(s, [](int64_t x) { return x == 0; }))
      return 0;
    if (isa<arith::AddIOp>(op))
      return s[0] + s[1];
    if (isa<arith::SubIOp>(op))
      return s[0] - s[1];
    if (isa<arith::MulIOp>(op) && (!s[0] || !s[1]))
      if (std::optional<int64_t> c =
              tt::intel::getFoldedConstantValue(op->getOperand(s[0] ? 1 : 0)))
        return (s[0] + s[1]) * *c;
    return std::nullopt;
  }
};

static std::optional<LoopSplit> planSplit(scf::ForOp forOp,
                                          ArrayRef<UnalignedWidthLoad> loads) {
  std::optional<int64_t> step =
      tt::intel::getFoldedConstantValue(forOp.getStep());
  if (!step || *step <= 0 || forOp.getUnsignedCmp())
    return std::nullopt;
  LoopSplit split(forOp, *step);
  for (UnalignedWidthLoad l : loads) {
    bool direct = true;
    for (Operation *p = l.load->getParentOp(); p != forOp; p = p->getParentOp())
      direct &= isa<scf::IfOp>(p);
    std::optional<int64_t> slope = split.slope(l.load.getIndices()[1]);
    if (direct && slope && *slope)
      split.entries.push_back({l, *slope});
  }
  if (split.entries.empty())
    return std::nullopt;
  return split;
}

static void execute(LoopSplit &split) {
  scf::ForOp forOp = split.forOp;
  int64_t step = split.step;
  Location loc = forOp.getLoc();
  ImplicitLocOpBuilder b(loc, forOp);
  ArithBuilder ab(b, loc);
  // From the live IR: an earlier split replaced the bounds and inits.
  split.first.map(forOp.getInductionVar(), forOp.getLowerBound());
  split.first.map(forOp.getRegionIterArgs(), forOp.getInitArgs());
  auto cst = [&](int64_t v) -> Value {
    return arith::ConstantIntOp::create(b, v, 64);
  };
  auto toI64 = [&](Value v) {
    return getValueOrCreateCastToIndexLike(b, loc, b.getI64Type(), v);
  };

  Value lb = toI64(forOp.getLowerBound());
  Value ub = toI64(forOp.getUpperBound());
  Value span = ab.sub(ub, lb);
  Value trips = arith::CeilDivSIOp::create(b, span, cst(step));
  Value zero = cst(0);
  Value n = arith::MaxSIOp::create(b, trips, zero);
  Value lo = n, hi = zero;
  for (auto &[l, d] : split.entries) {
    // Reaches iff prefix - tileCols < x0 + d * i < W (i32 wrap ignored).
    int64_t minX = l.prefix - l.load.getType().getShape()[1] + 1;
    int64_t maxX = l.width - 1;
    if (d < 0)
      std::swap(minX, maxX);
    Value x0 = toI64(split.firstIteration(b, l.load.getIndices()[1]));
    Value fromMin = ab.sub(cst(minX), x0);
    Value fromMax = ab.sub(cst(maxX), x0);
    Value firstIter = arith::CeilDivSIOp::create(b, fromMin, cst(d));
    Value lastIter = arith::FloorDivSIOp::create(b, fromMax, cst(d));
    Value first = arith::MaxSIOp::create(b, firstIter, zero);
    Value last = arith::MinSIOp::create(b, ab.add(lastIter, cst(1)), n);
    Value reaches = ab.slt(first, last);
    lo = arith::MinSIOp::create(b, lo, ab.select(reaches, first, n));
    hi = arith::MaxSIOp::create(b, hi, ab.select(reaches, last, zero));
  }
  hi = arith::MaxSIOp::create(b, hi, lo);

  // May exceed ub, unclamped: a clamp hides the alignment from block IO.
  Type ivTy = forOp.getLowerBound().getType();
  auto bound = [&](Value i) {
    return getValueOrCreateCastToIndexLike(b, loc, ivTy,
                                           ab.add(lb, ab.mul(i, cst(step))));
  };
  Value b1 = bound(lo), b2 = bound(hi);
  SmallVector<std::pair<unsigned, int64_t>> incs;
  for (BlockArgument arg : forOp.getRegionIterArgs())
    if (std::optional<int64_t> c = split.slope(arg))
      incs.push_back({arg.getArgNumber() - 1, *c});

  for (auto &[l, d] : split.entries)
    setMode(l.load, Mode::Unguarded);
  IRRewriter rewriter(b);
  auto firstTwo = splitForOpAtPoint(rewriter, forOp, b1);
  if (failed(firstTwo))
    return;
  auto lastTwo = splitForOpAtPoint(rewriter, firstTwo->second, b2);
  if (failed(lastTwo))
    return;
  scf::ForOp loops[3] = {firstTwo->first, lastTwo->first, lastTwo->second};
  // Closed-form inits: block IO marking cannot see through a loop result.
  b.setInsertionPoint(loops[0]);
  for (auto [idx, c] : incs) {
    Value init = loops[0].getInitArgs()[idx];
    Value init64 = toI64(init);
    for (auto [k, iters] : {std::pair(1, lo), std::pair(2, hi)}) {
      Value advanced = ab.add(init64, ab.mul(iters, cst(c)));
      loops[k].getInitArgsMutable()[idx].set(
          getValueOrCreateCastToIndexLike(b, loc, init.getType(), advanced));
    }
  }
  for (scf::ForOp loop : {loops[0], loops[2]})
    loop.walk([](tt::DescriptorLoadOp op) {
      if (op->hasAttr(kModeAttr) && getMode(op) == Mode::Unguarded)
        setMode(op, Mode::None);
    });
}

static void mergeDroppedColumns(UnalignedWidthLoad l, bool guarded) {
  tt::DescriptorLoadOp op = l.load;
  // Select at the DPAS operand, so the load is still rematerialized there.
  Value tile = op.getResult();
  for (Value v = tile; v.hasOneUse();) {
    auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*v.user_begin());
    if (!cvt)
      break;
    v = cvt.getResult();
    if (ttgi::hasDotDpasEncoding(cvt.getType())) {
      tile = v;
      break;
    }
  }
  bool feedsDpas = tile != op.getResult();
  SmallVector<OpOperand *> uses(llvm::make_pointer_range(tile.getUses()));
  auto tileTy = cast<RankedTensorType>(tile.getType());
  auto encoding = cast<ttg::DistributedEncodingTrait>(tileTy.getEncoding());
  Type elemTy = tileTy.getElementType();

  ImplicitLocOpBuilder b(op.getLoc(), op);
  ArithBuilder ab(b, op.getLoc());
  Type i32Ty = b.getI32Type(), i64Ty = b.getI64Type();
  auto cst = [&](int64_t v) -> Value {
    return arith::ConstantIntOp::create(b, v, 32);
  };
  auto splat = [&](Type ty, Value v) -> Value {
    return tt::SplatOp::create(b, ty, v);
  };
  auto zeros = [&](Type ty) -> Value {
    return arith::ConstantOp::create(b, b.getZeroAttr(ty));
  };
  auto range = [&](RankedTensorType ty, Value start) -> Value {
    Value iota = tt::MakeRangeOp::create(b, ty, 0, ty.getShape()[0]);
    Value starts = splat(ty, start);
    return ab.add(starts, iota);
  };
  auto slice = [&](unsigned dim, Type ty) {
    return RankedTensorType::get(
        {tileTy.getShape()[1 - dim]}, ty,
        ttg::SliceEncodingAttr::get(op.getContext(), dim, encoding));
  };

  Value row = op.getIndices()[0], col = op.getIndices()[1];
  RankedTensorType colTy = slice(1, elemTy);

  RankedTensorType rowsTy;
  Attribute loadEnc;
  Value inBounds, rowOffsets;
  auto prepare = [&] {
    if (feedsDpas)
      loadEnc = getLaneDistributedEncoding(colTy);
    rowsTy = (loadEnc ? colTy.cloneWithEncoding(loadEnc) : colTy).clone(i32Ty);
    Value rows = range(rowsTy, row);
    Value rowBound = splat(rowsTy, l.make.getShape()[0]);
    inBounds =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ult, rows, rowBound);
    Value stride = splat(rowsTy.clone(i64Ty), l.make.getStrides()[0]);
    Value rows64 = arith::ExtSIOp::create(b, rowsTy.clone(i64Ty), rows);
    rowOffsets = ab.mul(rows64, stride);
  };
  auto loadColumn = [&](int64_t k) -> Value {
    Value base = l.make.getBase();
    Value colBase =
        tt::AddPtrOp::create(b, base.getType(), base,
                             arith::ConstantIntOp::create(b, l.prefix + k, 64));
    Value ptrs = tt::AddPtrOp::create(
        b, rowsTy.clone(base.getType()),
        splat(rowsTy.clone(base.getType()), colBase), rowOffsets);
    Value other = zeros(rowsTy.clone(elemTy));
    Value column =
        tt::LoadOp::create(b, ptrs, inBounds, other, op.getCachePolicyAttr());
    return loadEnc ? ttg::ConvertLayoutOp::create(b, colTy, column) : column;
  };

  int64_t numDropped = l.width - l.prefix;
  SmallVector<Value> columns;
  // At the load, so no write in between is crossed.
  b.setInsertionPointAfter(op);
  if (guarded) {
    Value above = ab.sgt(col, cst(l.prefix - tileTy.getShape()[1]));
    Value below = ab.slt(col, cst(l.width));
    auto ifOp = scf::IfOp::create(
        b, ab._and(below, above),
        [&](OpBuilder &, Location) {
          prepare();
          SmallVector<Value> loaded;
          for (int64_t k = 0; k < numDropped; ++k)
            loaded.push_back(loadColumn(k));
          scf::YieldOp::create(b, loaded);
        },
        [&](OpBuilder &, Location) {
          scf::YieldOp::create(b, SmallVector<Value>(numDropped, zeros(colTy)));
        });
    columns = ifOp.getResults();
  } else {
    prepare();
    for (int64_t k = 0; k < numDropped; ++k)
      columns.push_back(loadColumn(k));
  }

  if (feedsDpas)
    b.setInsertionPointAfterValue(tile);
  Value tileCols = range(slice(0, i32Ty), col);
  Value patched = tile;
  auto patch = [&](int64_t k, Value column) {
    Value isCol = tt::BroadcastOp::create(
        b, tileTy.clone(b.getI1Type()),
        tt::ExpandDimsOp::create(
            b,
            arith::CmpIOp::create(b, arith::CmpIPredicate::eq, tileCols,
                                  splat(tileCols.getType(), cst(l.prefix + k))),
            0));
    Value vals = tt::BroadcastOp::create(
        b, tileTy, tt::ExpandDimsOp::create(b, column, 1));
    patched = ab.select(isCol, vals, patched);
  };

  for (int64_t k = 0; k < numDropped; ++k)
    patch(k, columns[k]);
  for (OpOperand *use : uses)
    use->set(patched);
}

struct TritonIntelGPUTailUnalignedWidthLoadsPass
    : public ttgi::impl::TritonIntelGPUTailUnalignedWidthLoadsBase<
          TritonIntelGPUTailUnalignedWidthLoadsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;
    tt::intel::ModuleAxisInfoAnalysis axisInfo(mod);
    // Only the index rule of MaterializeBlockPointer.
    llvm::MapVector<scf::ForOp, SmallVector<UnalignedWidthLoad>> loopLoads;
    mod.walk([&](tt::DescriptorLoadOp op) {
      std::optional<UnalignedWidthLoad> l = UnalignedWidthLoad::match(op);
      if (!l || !ttgi::isDescriptorIndexAligned(axisInfo, op.getIndices()[1],
                                                l->granule))
        return;
      setMode(op, Mode::Guarded);
      if (auto forOp = op->getParentOfType<scf::ForOp>())
        loopLoads[forOp].push_back(*l);
    });

    SmallVector<LoopSplit> splits;
    for (auto &[forOp, loads] : loopLoads)
      if (std::optional<LoopSplit> split = planSplit(forOp, loads))
        splits.push_back(std::move(*split));
    // Innermost only: splitting every level makes 3^depth clones.
    SmallPtrSet<Operation *, 4> outer;
    for (LoopSplit &split : splits)
      for (Operation *p = split.forOp->getParentOp(); p; p = p->getParentOp())
        outer.insert(p);
    for (LoopSplit &split : splits)
      if (!outer.contains(split.forOp))
        execute(split);

    SmallVector<tt::DescriptorLoadOp> tagged;
    mod.walk([&](tt::DescriptorLoadOp op) {
      if (op->hasAttr(kModeAttr))
        tagged.push_back(op);
    });
    DenseMap<Operation *, tt::MakeTensorDescOp> prefixDescs;
    for (tt::DescriptorLoadOp op : tagged) {
      Mode mode = getMode(op);
      op->removeAttr(kModeAttr);
      UnalignedWidthLoad l = *UnalignedWidthLoad::match(op);
      // Merging more columns under a branch in a loop spills.
      if (mode == Mode::Guarded && l.width - l.prefix > 1 &&
          op->getParentOfType<LoopLikeOpInterface>())
        continue;
      tt::MakeTensorDescOp &prefixDesc = prefixDescs[l.make];
      if (!prefixDesc) {
        OpBuilder b(l.make);
        b.setInsertionPointAfter(l.make);
        Value prefixWidth =
            arith::ConstantIntOp::create(b, l.make.getLoc(), l.prefix, 32);
        prefixDesc = cast<tt::MakeTensorDescOp>(b.clone(*l.make));
        prefixDesc.getShapeMutable()[1].set(prefixWidth);
      }
      op.getDescMutable().assign(prefixDesc);
      if (mode != Mode::None)
        mergeDroppedColumns(l, mode == Mode::Guarded);
    }
  }
};

} // namespace
