//===- SymbolicBoundsOracleTest.cpp ---------------------------------------===//
//
// An exhaustive model check of triton::intel::SymbolicBoundsProver at widths
// the machine can enumerate: i8 and i4.
//
// SymbolicBoundsTest.cpp pins *what the prover says*; this file checks that
// what it says is *true*. For each case it takes the verdict of one comparison,
// materializes the guard of a conditional verdict, and then executes the
// fixture for every value of every kernel argument with a width-exact
// interpreter - the oracle - comparing the verdict's claim against the actual
// mask elements:
//
//   Satisfied                every element of every iteration is true
//   Refuted                  every element of every iteration is false
//   ConditionallySatisfied   every element is true on every launch whose
//                            materialized guard is true
//   Unknown                  no claim, so nothing to check
//
// Launches the kernel contract excludes are skipped, not checked: the scf.for
// no-overflow condition and a false `llvm.intr.assume`. A launch
// that is skipped proves nothing, so the accounting below counts how many
// launches were valid, how many had a true guard, and how many mask elements
// were actually evaluated under a true guard; a case whose safety check never
// runs fails.
//
// Each case names exactly one verdict, so a prover that regresses to a vacuous
// Unknown fails here rather than silently stopping to prove anything. The
// oracle is deliberately independent of the prover: it knows the IR's
// wrapping semantics and nothing about affine forms.
//
// The last section checks coverage instead of safety: that the new guard
// holds wherever a legacy RemoveMasks guard soundly does.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/Range.h"
#include "intel/include/Analysis/SymbolicBounds.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Parser/Parser.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace mlir;
namespace tt = mlir::triton;

namespace {

/// Trip counts and width-fit tests are computed here, one step wider than any
/// value the IR can hold, so the check for "does this fit the IV's width" is
/// not itself subject to the wrapping it is testing. Arbitrary precision rather
/// than `__int128`, which MSVC does not provide.
class I128 {
public:
  I128(int64_t x = 0) : v(x) {}
  /// `a` read as signed.
  explicit I128(const APInt &a) : v(a) {}
  explicit operator int64_t() const { return static_cast<int64_t>(v); }

  /// The low 64 bits of the two's-complement pattern.
  uint64_t low64() const {
    if (v >= INT64_MIN && v <= INT64_MAX)
      return static_cast<uint64_t>(static_cast<int64_t>(v));
    static const DynamicAPInt two63 = DynamicAPInt(INT64_MAX) + 1;
    static const DynamicAPInt two64 = two63 * 2;
    DynamicAPInt r = mod(v, two64);
    if (r >= two63)
      r -= two64;
    return static_cast<uint64_t>(static_cast<int64_t>(r));
  }

  friend I128 operator+(const I128 &a, const I128 &b) { return a.v + b.v; }
  friend I128 operator-(const I128 &a, const I128 &b) { return a.v - b.v; }
  friend I128 operator*(const I128 &a, const I128 &b) { return a.v * b.v; }
  friend I128 operator/(const I128 &a, const I128 &b) { return a.v / b.v; }
  // DynamicAPInt's small-value path computes INT64_MIN % -1 natively, which
  // traps; x % -1 is 0 for every x.
  friend I128 operator%(const I128 &a, const I128 &b) {
    return b.v == -1 ? I128(0) : I128(a.v % b.v);
  }
  friend I128 operator<<(const I128 &a, unsigned s) {
    DynamicAPInt r = a.v;
    for (; s > 62; s -= 62)
      r *= int64_t(1) << 62;
    return r * (int64_t(1) << s);
  }
  I128 operator-() const { return -v; }
  I128 &operator++() {
    ++v;
    return *this;
  }
  friend bool operator==(const I128 &a, const I128 &b) { return a.v == b.v; }
  friend bool operator!=(const I128 &a, const I128 &b) { return a.v != b.v; }
  friend bool operator<(const I128 &a, const I128 &b) { return a.v < b.v; }
  friend bool operator<=(const I128 &a, const I128 &b) { return a.v <= b.v; }
  friend bool operator>(const I128 &a, const I128 &b) { return a.v > b.v; }
  friend bool operator>=(const I128 &a, const I128 &b) { return a.v >= b.v; }

private:
  I128(DynamicAPInt x) : v(std::move(x)) {}
  DynamicAPInt v;
};

//===----------------------------------------------------------------------===//
// Oracle
//===----------------------------------------------------------------------===//

/// One interpreter value: a scalar (`shape` empty, one element) or a row-major
/// tensor. Elements are APInts at the value's element width, so every result
/// wraps exactly where the IR's unflagged `arith` arithmetic wraps.
struct Val {
  SmallVector<int64_t, 4> shape;
  SmallVector<APInt, 4> elems;
};

std::string opName(Operation *op) { return op->getName().getStringRef().str(); }

/// The `w`-bit APInt holding the low `w` bits of `v`'s two's-complement
/// pattern. APInt's constructor asserts on a value that does not fit its
/// width, so an intended truncation has to say so.
APInt wrapToWidth(I128 v, unsigned w) {
  return APInt(w, v.low64(), /*isSigned=*/false, /*implicitTrunc=*/true);
}

Val scalarOf(APInt v) {
  Val r;
  r.elems.push_back(std::move(v));
  return r;
}

SmallVector<int64_t, 4> shapeOf(Type t) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    return SmallVector<int64_t, 4>(shaped.getShape());
  return {};
}

unsigned numElements(ArrayRef<int64_t> shape) {
  unsigned n = 1;
  for (int64_t d : shape)
    n *= static_cast<unsigned>(d);
  return n;
}

bool applyCmp(arith::CmpIPredicate pred, const APInt &a, const APInt &b) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return a == b;
  case arith::CmpIPredicate::ne:
    return a != b;
  case arith::CmpIPredicate::slt:
    return a.slt(b);
  case arith::CmpIPredicate::sle:
    return a.sle(b);
  case arith::CmpIPredicate::sgt:
    return a.sgt(b);
  case arith::CmpIPredicate::sge:
    return a.sge(b);
  case arith::CmpIPredicate::ult:
    return a.ult(b);
  case arith::CmpIPredicate::ule:
    return a.ule(b);
  case arith::CmpIPredicate::ugt:
    return a.ugt(b);
  case arith::CmpIPredicate::uge:
    return a.uge(b);
  }
  llvm_unreachable("unknown integer comparison predicate");
}

/// A width-exact interpreter for the fixture IR subset
/// (`arith.{constant,addi,subi,muli,divsi,remsi,extsi,extui,trunci,cmpi,andi,
/// xori,shrsi,select,mulsi_extended}`, `tt.{make_range,splat,expand_dims,
/// broadcast,get_program_id,return}`, `scf.for` with iter_args and
/// `unsignedCmp`, `scf.yield`, `llvm.intr.assume`), which is also exactly the
/// subset `materialize` emits, so the same interpreter evaluates guards,
/// including their checked i64 path.
class Oracle {
public:
  /// Executes one launch of the single `tt.func` in `m` with `argValues` bound
  /// to its arguments, and returns the results of the `arith.cmpi` carrying
  /// `loc("<maskName>")`: one entry per execution of that comparison, holding
  /// its per-element booleans. An empty result is a loop that ran zero times.
  ///
  /// Returns nullopt for a launch the kernel contract excludes - the scf.for
  /// no-overflow condition, a false `llvm.intr.assume`, or
  /// undefined arithmetic (division by zero, an out-of-range shift) - which
  /// the caller skips instead of checking. An operation outside the modelled
  /// subset fails the test rather than skipping silently.
  std::optional<std::vector<std::vector<bool>>>
  run(ModuleOp m, ArrayRef<int64_t> argValues, StringRef maskName) {
    env.clear();
    masks.clear();
    mask = maskName;
    targetLoop = nullptr;

    tt::FuncOp f;
    m.walk([&](tt::FuncOp op) { f = op; });
    if (!f) {
      ADD_FAILURE() << "oracle: no tt.func to run";
      return std::nullopt;
    }
    if (bindArgs(f, argValues) != Status::Ok)
      return std::nullopt;
    if (execBlock(f.getBody().front()) != Status::Ok)
      return std::nullopt;
    return masks;
  }

  /// Evaluates a materialized guard for one launch: a pure scalar expression
  /// over the kernel arguments, so it is evaluated on demand through its
  /// operand cone rather than by executing the function.
  bool evalGuard(Value guard, ArrayRef<int64_t> argValues) {
    env.clear();
    masks.clear();
    mask = StringRef();
    targetLoop = nullptr;

    if (!guard) {
      ADD_FAILURE() << "oracle: no guard to evaluate";
      return false;
    }
    Operation *def = guard.getDefiningOp();
    if (!def) {
      ADD_FAILURE() << "oracle: guard is a block argument";
      return false;
    }
    tt::FuncOp f = def->getParentOfType<tt::FuncOp>();
    if (!f) {
      ADD_FAILURE() << "oracle: guard is not inside a tt.func";
      return false;
    }
    if (bindArgs(f, argValues) != Status::Ok)
      return false;
    if (evalDemand(guard) != Status::Ok)
      return false;
    const Val &v = env.find(guard)->second;
    return !v.elems.empty() && v.elems.front().getBoolValue();
  }

  /// Executes one launch like `run`, and returns the trip count of every
  /// invocation of `target`, an scf.for, in execution order. A loop entered
  /// but empty still records a zero, so an enclosing loop that never runs is
  /// told apart from a nested loop that runs zero times. nullopt for a launch
  /// the contract excludes, as for `run`.
  std::optional<std::vector<uint64_t>>
  runTripCounts(ModuleOp m, ArrayRef<int64_t> argValues, scf::ForOp target) {
    env.clear();
    masks.clear();
    mask = StringRef();
    trips.clear();
    targetLoop = target.getOperation();

    tt::FuncOp f;
    m.walk([&](tt::FuncOp op) { f = op; });
    if (!f) {
      ADD_FAILURE() << "oracle: no tt.func to run";
      return std::nullopt;
    }
    std::optional<std::vector<uint64_t>> result;
    if (bindArgs(f, argValues) == Status::Ok &&
        execBlock(f.getBody().front()) == Status::Ok)
      result = trips;
    targetLoop = nullptr;
    return result;
  }

private:
  /// `Skip` ends the launch without a verdict (undefined or assume-violating);
  /// `Fail` means the oracle does not model something it met, and has already
  /// failed the test.
  enum class Status { Ok, Skip, Fail };

  /// A malformed fixture must fail rather than hang: no modelled loop over an
  /// i8 iteration space can run this many times.
  static constexpr int64_t kMaxIterations = 4096;

  Status bindArgs(tt::FuncOp f, ArrayRef<int64_t> argValues) {
    if (f.getNumArguments() != argValues.size()) {
      ADD_FAILURE() << "oracle: launch supplies " << argValues.size()
                    << " values for " << f.getNumArguments() << " arguments";
      return Status::Fail;
    }
    for (unsigned i = 0, e = f.getNumArguments(); i != e; ++i) {
      BlockArgument arg = f.getArgument(i);
      auto intTy = dyn_cast<IntegerType>(arg.getType());
      if (!intTy) {
        ADD_FAILURE() << "oracle: argument " << i << " is not an integer";
        return Status::Fail;
      }
      set(arg, scalarOf(wrapToWidth(argValues[i], intTy.getWidth())));
    }
    return Status::Ok;
  }

  /// Executes `block` up to its terminator, which the caller interprets.
  Status execBlock(Block &block) {
    for (Operation &op : block) {
      if (isa<tt::ReturnOp, scf::YieldOp>(&op))
        return Status::Ok;
      Status s = execOp(&op);
      if (s != Status::Ok)
        return s;
    }
    return Status::Ok;
  }

  Status execOp(Operation *op) {
    // One check for every handler: an operand must already have a value.
    for (Value operand : op->getOperands())
      if (!env.count(operand)) {
        ADD_FAILURE() << "oracle: operand of " << opName(op)
                      << " was never evaluated";
        return Status::Fail;
      }

    if (auto cst = dyn_cast<arith::ConstantOp>(op))
      return execConstant(cst);
    if (isa<arith::AddIOp>(op))
      return binary(op, [](const APInt &a, const APInt &b) {
        return std::optional<APInt>(a + b);
      });
    if (isa<arith::SubIOp>(op))
      return binary(op, [](const APInt &a, const APInt &b) {
        return std::optional<APInt>(a - b);
      });
    if (isa<arith::MulIOp>(op))
      return binary(op, [](const APInt &a, const APInt &b) {
        return std::optional<APInt>(a * b);
      });
    if (isa<arith::DivSIOp>(op))
      return binary(op,
                    [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                      // Both are undefined in LLVM IR, so the launch does not
                      // count.
                      if (b.isZero() || (a.isMinSignedValue() && b.isAllOnes()))
                        return std::nullopt;
                      return a.sdiv(b);
                    });
    if (isa<arith::RemSIOp>(op))
      return binary(op,
                    [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                      if (b.isZero() || (a.isMinSignedValue() && b.isAllOnes()))
                        return std::nullopt;
                      return a.srem(b);
                    });
    if (isa<arith::AndIOp>(op))
      return binary(op, [](const APInt &a, const APInt &b) {
        return std::optional<APInt>(a & b);
      });
    if (isa<arith::XOrIOp>(op))
      return binary(op, [](const APInt &a, const APInt &b) {
        return std::optional<APInt>(a ^ b);
      });
    if (isa<arith::ShRSIOp>(op))
      return binary(op,
                    [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                      // A shift at or past the width is poison, not a wrapped
                      // shift.
                      if (b.uge(a.getBitWidth()))
                        return std::nullopt;
                      return a.ashr(b.getZExtValue());
                    });
    if (auto ext = dyn_cast<arith::ExtSIOp>(op)) {
      unsigned w = elemWidth(ext.getType());
      return unary(
          op, [w](const APInt &a) { return std::optional<APInt>(a.sext(w)); });
    }
    if (auto ext = dyn_cast<arith::ExtUIOp>(op)) {
      unsigned w = elemWidth(ext.getType());
      return unary(
          op, [w](const APInt &a) { return std::optional<APInt>(a.zext(w)); });
    }
    if (auto trunc = dyn_cast<arith::TruncIOp>(op)) {
      unsigned w = elemWidth(trunc.getType());
      return unary(
          op, [w](const APInt &a) { return std::optional<APInt>(a.trunc(w)); });
    }
    if (auto mulx = dyn_cast<arith::MulSIExtendedOp>(op))
      return execMulSIExtended(mulx);
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      return execCmp(cmp);
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      return execSelect(sel);
    if (auto range = dyn_cast<tt::MakeRangeOp>(op))
      return execMakeRange(range);
    if (auto splat = dyn_cast<tt::SplatOp>(op))
      return execSplat(splat);
    if (auto expand = dyn_cast<tt::ExpandDimsOp>(op))
      return execExpandDims(expand);
    if (auto bcast = dyn_cast<tt::BroadcastOp>(op))
      return execBroadcast(bcast);
    if (auto pid = dyn_cast<tt::GetProgramIdOp>(op)) {
      // No case needs more than one program, and the launch space enumerated
      // here is over kernel arguments only, so a single program is modelled.
      set(pid.getResult(), scalarOf(APInt(elemWidth(pid.getType()), 0)));
      return Status::Ok;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      return execFor(forOp);
    if (auto assume = dyn_cast<LLVM::AssumeOp>(op)) {
      // The kernel contract excludes a launch that violates an assume.
      const Val &cond = env.find(assume.getCond())->second;
      if (cond.elems.empty() || !cond.elems.front().getBoolValue())
        return Status::Skip;
      return Status::Ok;
    }

    ADD_FAILURE() << "oracle: unmodelled operation " << opName(op);
    return Status::Fail;
  }

  Status execConstant(arith::ConstantOp cst) {
    Attribute attr = cst.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      set(cst.getResult(), scalarOf(intAttr.getValue()));
      return Status::Ok;
    }
    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
      if (!isa<IntegerType>(getElementTypeOrSelf(dense.getType()))) {
        ADD_FAILURE() << "oracle: non-integer dense constant";
        return Status::Fail;
      }
      Val out;
      out.shape = shapeOf(cst.getType());
      if (dense.isSplat()) {
        APInt v = dense.getSplatValue<APInt>();
        out.elems.assign(numElements(out.shape), v);
      } else {
        for (const APInt &v : dense.getValues<APInt>())
          out.elems.push_back(v);
      }
      set(cst.getResult(), std::move(out));
      return Status::Ok;
    }
    ADD_FAILURE() << "oracle: unmodelled constant attribute";
    return Status::Fail;
  }

  Status execMulSIExtended(arith::MulSIExtendedOp op) {
    const Val &a = env.find(op.getLhs())->second;
    const Val &b = env.find(op.getRhs())->second;
    unsigned w = elemWidth(op.getLow().getType());
    Val low, high;
    low.shape = high.shape = shapeOf(op.getLow().getType());
    for (unsigned i = 0, e = a.elems.size(); i != e; ++i) {
      // The full 2w-bit signed product, split into its two w-bit halves: the
      // multiply fits w bits exactly when `high` is `low`'s sign extension,
      // which is the test the checked guard path emits.
      APInt prod = a.elems[i].sext(2 * w) * b.elems[i].sext(2 * w);
      low.elems.push_back(prod.trunc(w));
      high.elems.push_back(prod.lshr(w).trunc(w));
    }
    set(op.getLow(), std::move(low));
    set(op.getHigh(), std::move(high));
    return Status::Ok;
  }

  Status execCmp(arith::CmpIOp op) {
    arith::CmpIPredicate pred = op.getPredicate();
    const Val &a = env.find(op.getLhs())->second;
    const Val &b = env.find(op.getRhs())->second;
    Val out;
    out.shape = shapeOf(op.getResult().getType());
    for (unsigned i = 0, e = a.elems.size(); i != e; ++i)
      out.elems.push_back(APInt(1, applyCmp(pred, a.elems[i], b.elems[i])));

    // The comparison under test: record this execution's elements.
    auto nameLoc = dyn_cast<NameLoc>(op->getLoc());
    if (!mask.empty() && nameLoc && nameLoc.getName() == mask) {
      std::vector<bool> bits;
      for (const APInt &e : out.elems)
        bits.push_back(e.getBoolValue());
      masks.push_back(std::move(bits));
    }
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execSelect(arith::SelectOp op) {
    const Val &c = env.find(op.getCondition())->second;
    const Val &t = env.find(op.getTrueValue())->second;
    const Val &f = env.find(op.getFalseValue())->second;
    Val out;
    out.shape = shapeOf(op.getResult().getType());
    for (unsigned i = 0, e = t.elems.size(); i != e; ++i) {
      // A scalar condition selects for every element.
      bool take = c.elems[c.elems.size() == 1 ? 0 : i].getBoolValue();
      out.elems.push_back(take ? t.elems[i] : f.elems[i]);
    }
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execMakeRange(tt::MakeRangeOp op) {
    Val out;
    out.shape = shapeOf(op.getType());
    unsigned w = elemWidth(op.getType());
    int64_t start = static_cast<int32_t>(op.getStart());
    for (unsigned i = 0, e = numElements(out.shape); i != e; ++i)
      out.elems.push_back(wrapToWidth(start + i, w));
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execSplat(tt::SplatOp op) {
    const Val &src = env.find(op.getSrc())->second;
    if (src.elems.size() != 1) {
      ADD_FAILURE() << "oracle: tt.splat of a non-scalar";
      return Status::Fail;
    }
    Val out;
    out.shape = shapeOf(op.getType());
    out.elems.assign(numElements(out.shape), src.elems.front());
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execExpandDims(tt::ExpandDimsOp op) {
    // Inserting a size-1 axis does not move any element in row-major order.
    Val out = env.find(op.getSrc())->second;
    out.shape = shapeOf(op.getType());
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execBroadcast(tt::BroadcastOp op) {
    const Val &src = env.find(op.getSrc())->second;
    SmallVector<int64_t, 4> outShape = shapeOf(op.getType());
    if (src.shape.size() != outShape.size()) {
      ADD_FAILURE() << "oracle: tt.broadcast changes rank";
      return Status::Fail;
    }
    Val out;
    out.shape = outShape;
    unsigned rank = outShape.size();
    SmallVector<int64_t, 4> idx(rank, 0);
    for (unsigned i = 0, e = numElements(outShape); i != e; ++i) {
      // Row-major: unrank the linear index, then read the source with every
      // size-1 axis pinned to 0.
      unsigned rest = i, srcLinear = 0;
      for (int d = rank - 1; d >= 0; --d) {
        idx[d] = rest % outShape[d];
        rest /= outShape[d];
      }
      for (unsigned d = 0; d != rank; ++d)
        srcLinear = srcLinear * src.shape[d] + (src.shape[d] == 1 ? 0 : idx[d]);
      out.elems.push_back(src.elems[srcLinear]);
    }
    set(op.getResult(), std::move(out));
    return Status::Ok;
  }

  Status execFor(scf::ForOp forOp) {
    Value iv = forOp.getInductionVar();
    auto ivTy = dyn_cast<IntegerType>(iv.getType());
    if (!ivTy) {
      ADD_FAILURE() << "oracle: scf.for induction variable is not an integer";
      return Status::Fail;
    }
    unsigned w = ivTy.getWidth();
    bool uns = forOp.getUnsignedCmp();

    auto read = [&](Value v) -> I128 {
      const APInt &a = env.find(v)->second.elems.front();
      return uns ? I128(a.zext(a.getBitWidth() + 1)) : I128(a);
    };
    I128 lb = read(forOp.getLowerBound());
    I128 ub = read(forOp.getUpperBound());
    I128 step = read(forOp.getStep());
    if (step <= 0) {
      ADD_FAILURE() << "oracle: scf.for step is not positive";
      return Status::Fail;
    }

    I128 n = ub > lb ? (ub - lb + step - 1) / step : 0;
    // The scf.for contract: the value one step past the last iteration
    // must be representable at the IV's width in the loop's own signedness, or
    // the loop is undefined and the launch does not count. The prover gets
    // this for free, so the oracle must grant exactly the same.
    I128 lo = uns ? I128(0) : -(I128(1) << (w - 1));
    I128 hi = uns ? (I128(1) << w) - 1 : (I128(1) << (w - 1)) - 1;
    I128 end = lb + n * step;
    if (end < lo || end > hi)
      return Status::Skip;
    if (n > kMaxIterations) {
      ADD_FAILURE() << "oracle: scf.for runs more than "
                    << static_cast<int64_t>(kMaxIterations) << " times";
      return Status::Fail;
    }
    if (forOp.getOperation() == targetLoop)
      trips.push_back(static_cast<uint64_t>(static_cast<int64_t>(n)));

    SmallVector<Val, 2> carried;
    for (Value init : forOp.getInitArgs())
      carried.push_back(env.find(init)->second);

    Block *body = forOp.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    for (I128 j = 0; j < n; ++j) {
      set(iv, scalarOf(wrapToWidth(lb + j * step, w)));
      for (unsigned i = 0, e = carried.size(); i != e; ++i)
        set(forOp.getRegionIterArgs()[i], carried[i]);
      Status s = execBlock(*body);
      if (s != Status::Ok)
        return s;
      SmallVector<Val, 2> next;
      for (Value v : yield.getOperands())
        next.push_back(env.find(v)->second);
      carried = std::move(next);
    }
    // A zero-trip loop yields its initial values.
    for (unsigned i = 0, e = forOp.getNumResults(); i != e; ++i)
      set(forOp.getResult(i), carried[i]);
    return Status::Ok;
  }

  /// Evaluates `v` and everything it reads, for the guard paths.
  Status evalDemand(Value v) {
    if (env.count(v))
      return Status::Ok;
    Operation *def = v.getDefiningOp();
    if (!def) {
      ADD_FAILURE() << "oracle: guard reads an unbound block argument";
      return Status::Fail;
    }
    if (def->getNumRegions() != 0) {
      ADD_FAILURE() << "oracle: guard reads the result of " << opName(def);
      return Status::Fail;
    }
    for (Value operand : def->getOperands()) {
      Status s = evalDemand(operand);
      if (s != Status::Ok)
        return s;
    }
    return execOp(def);
  }

  using BinFn =
      llvm::function_ref<std::optional<APInt>(const APInt &, const APInt &)>;
  using UnFn = llvm::function_ref<std::optional<APInt>(const APInt &)>;

  Status binary(Operation *op, BinFn f) {
    const Val &a = env.find(op->getOperand(0))->second;
    const Val &b = env.find(op->getOperand(1))->second;
    Val out;
    out.shape = shapeOf(op->getResult(0).getType());
    for (unsigned i = 0, e = a.elems.size(); i != e; ++i) {
      std::optional<APInt> r = f(a.elems[i], b.elems[i]);
      if (!r)
        return Status::Skip;
      out.elems.push_back(*r);
    }
    set(op->getResult(0), std::move(out));
    return Status::Ok;
  }

  Status unary(Operation *op, UnFn f) {
    const Val &a = env.find(op->getOperand(0))->second;
    Val out;
    out.shape = shapeOf(op->getResult(0).getType());
    for (const APInt &e : a.elems) {
      std::optional<APInt> r = f(e);
      if (!r)
        return Status::Skip;
      out.elems.push_back(*r);
    }
    set(op->getResult(0), std::move(out));
    return Status::Ok;
  }

  static unsigned elemWidth(Type t) {
    return cast<IntegerType>(getElementTypeOrSelf(t)).getWidth();
  }

  /// Inserts after every read of the operands, so no reference into `env` is
  /// live across the insertion.
  void set(Value v, Val val) { env[v] = std::move(val); }

  DenseMap<Value, Val> env;
  StringRef mask;
  std::vector<std::vector<bool>> masks;
  /// `runTripCounts`: the loop being measured, and its trip counts so far.
  Operation *targetLoop = nullptr;
  std::vector<uint64_t> trips;
};

/// The total number of mask elements `run` evaluated: the per-iteration,
/// per-element count, and so 0 for a loop that ran zero times.
unsigned countElements(const std::vector<std::vector<bool>> &run) {
  unsigned n = 0;
  for (const std::vector<bool> &iteration : run)
    n += iteration.size();
  return n;
}

/// True when every element of every iteration equals `value`. Vacuously true
/// for a zero-trip loop, which makes no claim either way.
bool every(const std::vector<std::vector<bool>> &run, bool value) {
  for (const std::vector<bool> &iteration : run)
    for (bool b : iteration)
      if (b != value)
        return false;
  return true;
}

/// One launch's arguments: signed representatives at each argument's width, in
/// argument order. Printable, so a failing launch names itself.
struct LaunchArgs {
  SmallVector<int64_t, 2> values;
  operator ArrayRef<int64_t>() const { return values; }
};

std::ostream &operator<<(std::ostream &os, const LaunchArgs &args) {
  os << " args(";
  for (unsigned i = 0, e = args.values.size(); i != e; ++i)
    os << (i ? ", " : "") << args.values[i];
  return os << ")";
}

/// Every combination of the function's argument values, each argument
/// enumerated over the whole of its width. A function with no arguments has
/// exactly one launch, not none.
std::vector<LaunchArgs> allArgumentValues(ModuleOp m) {
  tt::FuncOp f;
  m.walk([&](tt::FuncOp op) { f = op; });
  if (!f) {
    ADD_FAILURE() << "no tt.func to enumerate";
    return {};
  }

  // i4 and i8 arguments only: anything wider does not enumerate.
  SmallVector<unsigned, 2> widths;
  for (unsigned i = 0, e = f.getNumArguments(); i != e; ++i) {
    auto intTy = dyn_cast<IntegerType>(f.getArgument(i).getType());
    if (!intTy) {
      ADD_FAILURE() << "argument " << i << " is not an integer, so the "
                    << "launch space cannot be enumerated";
      return {};
    }
    if (intTy.getWidth() > 8) {
      ADD_FAILURE() << "argument " << i << " is " << intTy.getWidth()
                    << " bits wide; the oracle enumerates i4 and i8 only";
      return {};
    }
    widths.push_back(intTy.getWidth());
  }

  std::vector<LaunchArgs> out{LaunchArgs{}};
  for (unsigned w : widths) {
    int64_t lo = -(int64_t(1) << (w - 1)), hi = (int64_t(1) << (w - 1)) - 1;
    std::vector<LaunchArgs> grown;
    grown.reserve(out.size() * (hi - lo + 1));
    for (const LaunchArgs &prefix : out)
      for (int64_t v = lo; v <= hi; ++v) {
        LaunchArgs next = prefix;
        next.values.push_back(v);
        grown.push_back(std::move(next));
      }
    out = std::move(grown);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Fixture
//===----------------------------------------------------------------------===//

/// The SymbolicBoundsTest scaffolding, plus the oracle and the guard
/// materialization the model check needs.
class OracleFixture : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<tt::TritonDialect>();
    ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  }

  /// Parses `ir`, runs the range analysis over it exactly as a consumer pass
  /// does, and builds the prover. Must be called before any query.
  void parse(StringRef ir) {
    module = parseSourceString<ModuleOp>(ir, &ctx);
    ASSERT_TRUE(module) << "failed to parse:\n" << ir.str();
    ModuleOp mod = module.get();
    domInfo = std::make_unique<DominanceInfo>(mod);
    solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(mod, *domInfo);
    ASSERT_TRUE(succeeded(solver->initializeAndRun(mod)));
    prover = std::make_unique<tt::intel::SymbolicBoundsProver>(*solver,
                                                               *domInfo, mod);
  }

  /// The single function in the parsed IR. A ModuleOp body's front() is that
  /// operation, which has no getBody(), so walk to it as arg() does.
  tt::FuncOp func() {
    tt::FuncOp f;
    module->walk([&](tt::FuncOp op) { f = op; });
    return f;
  }

  /// The query context for a value: its defining operation, and the innermost
  /// enclosing scf.for if any.
  tt::intel::QueryContext at(Value v) {
    Operation *def = v.getDefiningOp();
    return {def ? def : &func().getBody().front().front(),
            def ? def->getParentOfType<scf::ForOp>() : nullptr};
  }

  /// Returns the single result of the operation carrying `loc("<name>")`, so
  /// the tests do not depend on the SSA numbering the parser assigns.
  Value get(StringRef name) {
    Value found;
    unsigned matches = 0;
    module->walk([&](Operation *op) {
      auto nameLoc = dyn_cast<NameLoc>(op->getLoc());
      if (!nameLoc || nameLoc.getName() != name)
        return;
      ++matches;
      if (op->getNumResults() == 1)
        found = op->getResult(0);
    });
    EXPECT_EQ(matches, 1u) << "expected exactly one op named '" << name.str()
                           << "'";
    EXPECT_TRUE(found) << "op named '" << name.str()
                       << "' has no single result";
    return found;
  }

  /// Returns the scf.for carrying `loc("<name>")`.
  scf::ForOp forNamed(StringRef name) {
    scf::ForOp found;
    unsigned matches = 0;
    module->walk([&](scf::ForOp op) {
      auto nameLoc = dyn_cast<NameLoc>(op.getLoc());
      if (!nameLoc || nameLoc.getName() != name)
        return;
      ++matches;
      found = op;
    });
    EXPECT_EQ(matches, 1u) << "expected exactly one loop named '" << name.str()
                           << "'";
    return found;
  }

  /// Materializes `p`'s conditions as one i1 guard immediately before `loop`.
  Value materializeBeforeLoop(const tt::intel::BoundProof &p, scf::ForOp loop) {
    OpBuilder b(loop);
    return tt::intel::materialize(p.conditions, loop.getOperation(), b);
  }

  /// Returns argument `idx` of the (single) function in the parsed IR.
  Value arg(unsigned idx) {
    Value found;
    module->walk([&](tt::FuncOp funcOp) { found = funcOp.getArgument(idx); });
    EXPECT_TRUE(found) << "no function argument " << idx;
    return found;
  }

  /// The proof of the comparison named `loc("<name>")`.
  tt::intel::BoundProof proof(Value cmp) {
    auto op = cast<arith::CmpIOp>(cmp.getDefiningOp());
    return prover->prove(op.getPredicate(), op.getLhs(), op.getRhs(), at(cmp));
  }

  /// Materializes `p`'s conditions as one i1 guard, immediately before the
  /// comparison's enclosing scf.for, or before the comparison itself when
  /// there is no loop. Both points are dominated by every loop-invariant
  /// symbol a condition can name.
  Value materializeAtQuery(const tt::intel::BoundProof &p, Value cmp) {
    Operation *cmpOp = cmp.getDefiningOp();
    auto loop = cmpOp->getParentOfType<scf::ForOp>();
    Operation *before = loop ? loop.getOperation() : cmpOp;
    OpBuilder b(before);
    return tt::intel::materialize(p.conditions, before, b);
  }

protected:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<DataFlowSolver> solver;
  std::unique_ptr<tt::intel::SymbolicBoundsProver> prover;
  Oracle oracle;
};

//===----------------------------------------------------------------------===//
// Fixtures
//===----------------------------------------------------------------------===//
//
// Scalars and loop bounds are i8, or i4 where the case is scalar-only, and are
// sign-extended to i32 where they meet a lane, because tt.make_range requires
// i32 elements. Loop bounds come from i8 values directly and are never scaled,
// so every launch runs a bounded number of iterations. The comment on each
// fixture is its expected verdict.

static const char *kSameLaneTwoAxes = R"(   // Unknown (verdict-only)
  tt.func @f() {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %col = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %cb = tt.broadcast %col : tensor<4x1xi32> -> tensor<4x4xi32>
    %rb = tt.broadcast %row : tensor<1x4xi32> -> tensor<4x4xi32>
    %cmp = arith.cmpi slt, %cb, %rb : tensor<4x4xi32> loc("cmp")
    tt.return
  })";
static const char *kNarrowRangeWrapI8 =
    R"( // Conditional{arg0 <= 27}: the open wrap guard
  tt.func @f(%a: i8) {
    %c100 = arith.constant 100 : i8
    %c110 = arith.constant 110 : i8
    %ge = arith.cmpi sge, %a, %c100 : i8
    llvm.intr.assume %ge : i1
    %le = arith.cmpi sle, %a, %c110 : i8
    llvm.intr.assume %le : i1
    %y = arith.addi %a, %c100 : i8
    %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
    tt.return
  })";
static const char *kOffsetIterArgI8 =
    R"(   // Unknown: constant wrap 123 + 5 > 127
  tt.func @f() {
    %c1 = arith.constant 1 : i8
    %c5 = arith.constant 5 : i8
    %c0 = arith.constant 0 : i8
    %lb = arith.constant 120 : i8
    %ub = arith.constant 124 : i8
    %init = arith.addi %lb, %c5 : i8
    scf.for %i = %lb to %ub step %c1 iter_args(%o = %init) -> (i8) : i8 {
      %cmp = arith.cmpi sge, %o, %c0 : i8 loc("cmp")
      %next = arith.addi %o, %c1 : i8
      scf.yield %next : i8
    }
    tt.return
  })";
static std::string depthCapChainI8() { // Unknown: depth budget exhausted
  std::string ir = "tt.func @f(%a: i8) {\n  %c1 = arith.constant 1 : i32\n"
                   "  %x = arith.extsi %a : i8 to i32\n";
  for (int i = 1; i <= 20; ++i)
    ir += "  %v" + std::to_string(i) + " = arith.addi %" +
          (i == 1 ? std::string("x") : "v" + std::to_string(i - 1)) +
          ", %c1 : i32\n";
  return ir + "  %cmp = arith.cmpi slt, %v20, %x : i32 loc(\"cmp\")\n"
              "  tt.return\n}\n";
}
static const std::string kDepthCapChainI8 = depthCapChainI8();
static const char *kUnsignedRefuteEdgeI8 =
    R"( // Unknown: ub = 132 (bit pattern) fails NonNegative(ub)
  tt.func @f() {
    %c4 = arith.constant 4 : i8
    %c120 = arith.constant 120 : i8
    %lb = arith.constant 120 : i8
    %ub = arith.constant -124 : i8               // 132 as unsigned
    scf.for unsigned %i = %lb to %ub step %c4 : i8 {   // IVs 120, 124, 128 (= -128)
      %mask = arith.cmpi slt, %i, %c120 : i8 loc("mask")   // true at the third IV
      scf.yield
    }
    tt.return
  })";
static const char *kConstantRefutedI8 =
    R"(   // Refuted: idx <= 63 < 100 on every launch
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c16 = arith.constant 16 : i8
    %c64 = arith.constant 64 : i8
    %c100 = arith.constant dense<100> : tensor<16xi32>
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    scf.for %i = %c0 to %c64 step %c16 : i8 {
      %i32 = arith.extsi %i : i8 to i32
      %is = tt.splat %i32 : i32 -> tensor<16xi32>
      %idx = arith.addi %is, %lane : tensor<16xi32>
      %mask = arith.cmpi sge, %idx, %c100 : tensor<16xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kLoopBoundWrapI8 =
    R"(   // Conditional{arg0 >= -127}: as LoopBoundWrapObligation
  tt.func @f(%n: i8) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %ub = arith.subi %n, %c1 : i8
    scf.for %iv = %c0 to %ub step %c1 : i8 {
      %mask = arith.cmpi slt, %iv, %n : i8 loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kReductionLoopI8 =
    R"(      // Conditional{arg0 divisible by 4}; 256 launches
  tt.func @reduction_loop(%n8: i8) {
    %c0 = arith.constant 0 : i8
    %c4 = arith.constant 4 : i8
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %n = arith.extsi %n8 : i8 to i32
    %ns = tt.splat %n : i32 -> tensor<4xi32>
    scf.for %r = %c0 to %n8 step %c4 : i8 {
      %r32 = arith.extsi %r : i8 to i32
      %rs = tt.splat %r32 : i32 -> tensor<4xi32>
      %idx = arith.addi %rs, %lane : tensor<4xi32>
      %mask = arith.cmpi slt, %idx, %ns : tensor<4xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kCdivKLoopI8 =
    R"(      // Conditional{arg0 divisible by 4; arg0 >= 0; arg0 <= 124}
  tt.func @cdiv_k_loop(%K: i8) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c3 = arith.constant 3 : i8
    %c4 = arith.constant 4 : i8
    %num = arith.addi %K, %c3 : i8
    %q = arith.divsi %num, %c4 : i8
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    scf.for %k = %c0 to %q step %c1 : i8 {
      %k4 = arith.muli %k, %c4 : i8
      %rem = arith.subi %K, %k4 : i8
      %rem32 = arith.extsi %rem : i8 to i32
      %rs = tt.splat %rem32 : i32 -> tensor<4xi32>
      %mask = arith.cmpi slt, %lane, %rs : tensor<4xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kSameLoadTwoAxes = R"(   // Unknown (verdict-only)
  tt.func @f(%p: !tt.ptr<i32>) {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %pp = tt.addptr %ps, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %t = tt.load %pp : tensor<4x!tt.ptr<i32>>
    %col = tt.expand_dims %t {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row = tt.expand_dims %t {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %cb = tt.broadcast %col : tensor<4x1xi32> -> tensor<4x4xi32>
    %rb = tt.broadcast %row : tensor<1x4xi32> -> tensor<4x4xi32>
    %cmp = arith.cmpi slt, %cb, %rb : tensor<4x4xi32> loc("cmp")
    tt.return
  })";
static const char *kScalarWrapI4 = R"(      // Conditional{arg0 <= 6}
  tt.func @f(%x: i4) {
    %c1 = arith.constant 1 : i4
    %y = arith.addi %x, %c1 : i4
    %cmp = arith.cmpi sgt, %y, %x : i4 loc("cmp")
    tt.return
  })";
static const char *kUnsignedLoopI8 =
    R"(    // Conditional{arg0 >= 0; arg1 >= 0; arg1 <= 124}
  tt.func @f(%lb: i8, %ub: i8) {
    %c2 = arith.constant 2 : i8
    %c4 = arith.constant 4 : i8
    scf.for unsigned %i = %lb to %ub step %c4 : i8 {
      %j = arith.addi %i, %c2 : i8
      %mask = arith.cmpi sgt, %j, %i : i8 loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kUnsignedPredI8 =
    R"(    // Conditional{arg0 <= 99; arg0 >= 0}: residual guard, then ult obligation
  tt.func @f(%x: i8) {
    %c100 = arith.constant 100 : i8
    %cmp = arith.cmpi ult, %x, %c100 : i8 loc("cmp")
    tt.return
  })";
static const char *kExtUII8 = R"(           // Conditional{arg0 >= 0}
  tt.func @f(%a: i8) {
    %c0 = arith.constant 0 : i32
    %e = arith.extui %a : i8 to i32
    %cmp = arith.cmpi sge, %e, %c0 : i32 loc("cmp")
    tt.return
  })";
static const char *kEmptyLoopI8 =
    R"(       // Conditional{arg1 - arg0 divisible by 4}; lb >= ub runs zero times
  tt.func @f(%lb8: i8, %ub8: i8) {
    %c4 = arith.constant 4 : i8
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %ub = arith.extsi %ub8 : i8 to i32
    %ubs = tt.splat %ub : i32 -> tensor<4xi32>
    scf.for %r = %lb8 to %ub8 step %c4 : i8 {
      %r32 = arith.extsi %r : i8 to i32
      %rs = tt.splat %r32 : i32 -> tensor<4xi32>
      %idx = arith.addi %rs, %lane : tensor<4xi32>
      %mask = arith.cmpi slt, %idx, %ubs : tensor<4xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";
static const char *kProductSignI8 = R"(     // Conditional{p >= 0}
  tt.func @f(%a: i8, %b: i8) {
    %c0 = arith.constant 0 : i8
    %p = arith.muli %a, %b : i8 loc("p")
    %cmp = arith.cmpi sge, %p, %c0 : i8 loc("cmp")
    tt.return
  })";
static const char *kProductSignI4 = R"(     // Conditional{p >= 0}
  tt.func @f(%a: i4, %b: i4) {
    %c0 = arith.constant 0 : i4
    %p = arith.muli %a, %b : i4 loc("p")
    %cmp = arith.cmpi sge, %p, %c0 : i4 loc("cmp")
    tt.return
  })";

//===----------------------------------------------------------------------===//
// Case table
//===----------------------------------------------------------------------===//

using V = tt::intel::BoundProof::Verdict;
// Each case names exactly one verdict, so a vacuous Unknown fails.
// Safety is then checked by execution for every verdict except Unknown, which
// makes no claim; Unknown cases are verdict-only, so the oracle needs no
// tt.load support for two_axes_load.
// `search` is the candidates the case needs: "loop" cases use only the
// exact-loop-end and exact-cdiv candidates and obligation guards; "residual"
// cases need a term-sign, quotient-threshold or residual-guard candidate.
struct Case {
  const char *name;
  std::string ir;
  const char *mask;
  std::vector<V> allowed;
  const char *search = "loop";
  bool vacuous = false;
};
static const Case kCases[] = {
    {"reduction_i8", kReductionLoopI8, "mask", {V::ConditionallySatisfied}},
    {"cdiv_k_i8", kCdivKLoopI8, "mask", {V::ConditionallySatisfied}},
    {"two_axes", kSameLaneTwoAxes, "cmp", {V::Unknown}},
    {"two_axes_load",
     kSameLoadTwoAxes,
     "cmp",
     {V::Unknown}}, // loaded tensor on two axes
    {"wrap_i8",
     kNarrowRangeWrapI8,
     "cmp",
     {V::ConditionallySatisfied},
     "loop",
     /*vacuous=*/true}, // guard arg0 <= 27 never holds under the assumes
    {"wrap_i4",
     kScalarWrapI4,
     "cmp",
     {V::ConditionallySatisfied}}, // x + 1 > x, guard x <= 6
    {"unsigned_edge", kUnsignedLoopI8, "mask", {V::ConditionallySatisfied}},
    {"unsigned_pred",
     kUnsignedPredI8,
     "cmp",
     {V::ConditionallySatisfied},
     "residual"}, // residual guard, then the ult obligations
    {"extui",
     kExtUII8,
     "cmp",
     {V::ConditionallySatisfied},
     "residual"}, // term sign proves the operand sign; the obligation alone is
                  // not search evidence
    {"zero_trip", kEmptyLoopI8, "mask", {V::ConditionallySatisfied}},
    {"product_wrap",
     kProductSignI8,
     "cmp",
     {V::ConditionallySatisfied},
     "residual"}, // term sign guards p itself
    {"product_wrap_i4",
     kProductSignI4,
     "cmp",
     {V::ConditionallySatisfied},
     "residual"},
    {"offset_iter_arg", kOffsetIterArgI8, "cmp", {V::Unknown}},
    {"depth_cap", kDepthCapChainI8, "cmp", {V::Unknown}}, // budget exhausted
    {"loop_bound_wrap",
     kLoopBoundWrapI8,
     "mask",
     {V::ConditionallySatisfied}}, // guard arg0 >= -127
    {"unsigned_refute",
     kUnsignedRefuteEdgeI8,
     "mask",
     {V::Unknown}}, // never Refuted: IV 128 reads as -128
    {"const_refuted",
     kConstantRefutedI8,
     "mask",
     {V::Refuted}}, // exercises Refuted => all false
};

class SymbolicBoundsOracleTest : public OracleFixture,
                                 public ::testing::WithParamInterface<Case> {};

TEST_P(SymbolicBoundsOracleTest, VerdictMatchesExhaustiveExecution) {
  using tt::intel::BoundProof;
  const Case &c = GetParam();
  parse(c.ir);
  BoundProof p = proof(get(c.mask));
  EXPECT_TRUE(llvm::is_contained(c.allowed, p.verdict))
      << c.name << ": " << toString(p);
  if (p.verdict == BoundProof::Unknown)
    return;
  // Guards go immediately before the comparison's enclosing scf.for, or before
  // the comparison itself in loop-free fixtures.
  Value guard = p.verdict == BoundProof::ConditionallySatisfied
                    ? materializeAtQuery(p, get(c.mask))
                    : Value();
  unsigned valid = 0, guardTrue = 0,
           guardedElems = 0; // a safety check that never runs proves nothing
  for (auto args : allArgumentValues(*module)) { // every combination of the
                                                 // tt.func's i4/i8 arguments
    auto run = oracle.run(module.get(), args, c.mask);
    if (!run)
      continue; // UB or assume-violating launch
    ++valid;
    if (guard && oracle.evalGuard(guard, args)) {
      ++guardTrue;
      guardedElems += countElements(*run); // zero for a launch whose loop is
                                           // empty
    }
    bool allTrue = every(*run, true), allFalse = every(*run, false);
    switch (p.verdict) {
    case BoundProof::Satisfied:
      EXPECT_TRUE(allTrue) << c.name << args;
      break;
    case BoundProof::Refuted:
      EXPECT_TRUE(allFalse) << c.name << args;
      break;
    case BoundProof::ConditionallySatisfied:
      if (oracle.evalGuard(guard, args))
        EXPECT_TRUE(allTrue) << c.name << args;
      break;
    case BoundProof::Unknown:
      break;
    }
  }
  EXPECT_GT(valid, 0u) << c.name;
  if (guard && !c.vacuous) // guard-true launches with empty loops do not count
    EXPECT_GT(guardedElems, 0u)
        << c.name << ": no guarded mask element was ever evaluated";
  RecordProperty(std::string(c.name) + "_valid", valid);
  RecordProperty(std::string(c.name) + "_guard_true", guardTrue);
  RecordProperty(std::string(c.name) + "_guarded_elements", guardedElems);
}

// Two suites, so --gtest_filter can select cases by the candidates they need,
// e.g. `--gtest_filter='LoopCandidates/*'`. `search` is the single source of
// both lists, so a case cannot be in neither or both.
static std::vector<Case> casesFor(StringRef search) {
  std::vector<Case> out;
  for (const Case &c : kCases)
    if (c.search == search)
      out.push_back(c);
  return out;
}

/// Names each instantiation after its case, so a failure reads as
/// `<suite>/SymbolicBoundsOracleTest.VerdictMatchesExhaustiveExecution/<case>`.
static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(LoopCandidates, SymbolicBoundsOracleTest,
                         ::testing::ValuesIn(casesFor("loop")), caseName);
INSTANTIATE_TEST_SUITE_P(ResidualCandidates, SymbolicBoundsOracleTest,
                         ::testing::ValuesIn(casesFor("residual")), caseName);

//===----------------------------------------------------------------------===//
// Trip counts
//===----------------------------------------------------------------------===//
//
// `tripCountAtLeast` is checked against the trip count of every invocation of
// the queried loop, including a nested loop entered many times and one that
// runs zero times. Satisfied must hold in every invocation of every valid
// launch, Refuted must fail in every one, and a Conditional guard - evaluated
// over the kernel arguments only, so a fixture whose guard needs an enclosing
// IV fails loudly - must imply it. Nested fixtures use i4 arguments: the
// enumeration is Cartesian and each launch interprets every iteration.

struct TripOracleCase {
  const char *name;
  std::string ir;
  int64_t n;
  std::vector<V> allowed;
  bool vacuous = false; // waive the "some guard-true invocation ran" check
};

static const TripOracleCase kTripCases[] = {
    {"const_single_trip_n1",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c64 = arith.constant 64 : i8
    scf.for %i = %c0 to %c64 step %c64 : i8 {
    } loc("L")
    tt.return
  })",
     1,
     {V::Satisfied}},
    // 0 + 1 * 64 fits in i8 but 0 + 2 * 64 does not: the contract is about the
    // actual trip count, so the n = 2 query is still answered.
    {"const_single_trip_n2",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c64 = arith.constant 64 : i8
    scf.for %i = %c0 to %c64 step %c64 : i8 {
    } loc("L")
    tt.return
  })",
     2,
     {V::Refuted}},
    {"empty_loop_n1",
     R"(
  tt.func @f() {
    %c5 = arith.constant 5 : i8
    %c3 = arith.constant 3 : i8
    %c1 = arith.constant 1 : i8
    scf.for %i = %c5 to %c3 step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     1,
     {V::Refuted}},
    {"dynamic_step_one_n2",
     R"(
  tt.func @f(%N: i8) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    scf.for %i = %c0 to %N step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     2,
     {V::ConditionallySatisfied}},
    // Step 32: with step 64 every launch that satisfies the guard would leave
    // the contract (0 + 2 * 64 does not fit i8) and be skipped.
    {"dynamic_step_32_n2",
     R"(
  tt.func @f(%N: i8) {
    %c0 = arith.constant 0 : i8
    %c32 = arith.constant 32 : i8
    scf.for %i = %c0 to %N step %c32 : i8 {
    } loc("L")
    tt.return
  })",
     2,
     {V::ConditionallySatisfied}},
    {"unsigned_dynamic_n2",
     R"(
  tt.func @f(%N: i8) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    scf.for unsigned %i = %c0 to %N step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     2,
     {V::ConditionallySatisfied}},
    {"unsigned_two_args_n1",
     R"(
  tt.func @f(%M: i4, %N: i4) {
    %c1 = arith.constant 1 : i4
    scf.for unsigned %i = %M to %N step %c1 : i4 {
    } loc("L")
    tt.return
  })",
     1,
     {V::ConditionallySatisfied}},
    // The guard is necessary: x = 127 wraps x + 1 to -128, an empty loop.
    {"raw_plus_one_n1",
     R"(
  tt.func @f(%x: i8) {
    %c1 = arith.constant 1 : i8
    %xp1 = arith.addi %x, %c1 : i8
    scf.for %j = %x to %xp1 step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     1,
     {V::ConditionallySatisfied}},
    {"masked_plus_one_n1",
     R"(
  tt.func @f(%a: i8) {
    %c1 = arith.constant 1 : i8
    %c63 = arith.constant 63 : i8
    %x = arith.andi %a, %c63 : i8
    %xp1 = arith.addi %x, %c1 : i8
    scf.for %j = %x to %xp1 step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     1,
     {V::Satisfied}},
    {"masked_plus_one_n2",
     R"(
  tt.func @f(%a: i8) {
    %c1 = arith.constant 1 : i8
    %c63 = arith.constant 63 : i8
    %x = arith.andi %a, %c63 : i8
    %xp1 = arith.addi %x, %c1 : i8
    scf.for %j = %x to %xp1 step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     2,
     {V::Refuted}},
    {"upper_bound_minus_one_n1",
     R"(
  tt.func @f(%N: i8) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %nm1 = arith.subi %N, %c1 : i8
    scf.for %i = %c0 to %nm1 step %c1 : i8 {
    } loc("L")
    tt.return
  })",
     1,
     {V::ConditionallySatisfied}},
    // Inner 0..i under outer 0..2 is entered twice, with trip counts 0 and 1.
    {"nested_const_outer_n2",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c2 = arith.constant 2 : i8
    scf.for %i = %c0 to %c2 step %c1 : i8 {
      scf.for %j = %c0 to %i step %c1 : i8 {
      } loc("L")
    }
    tt.return
  })",
     2,
     {V::Refuted}},
    // Inner 0..i under outer 0..3 can reach two trips: n = 2 is not refuted
    // (an Unknown verdict is pinned, not executed), n = 3 is.
    {"nested_boundary_outer_n2",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c3 = arith.constant 3 : i8
    scf.for %i = %c0 to %c3 step %c1 : i8 {
      scf.for %j = %c0 to %i step %c1 : i8 {
      } loc("L")
    }
    tt.return
  })",
     2,
     {V::Unknown}},
    {"nested_boundary_outer_n3",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c3 = arith.constant 3 : i8
    scf.for %i = %c0 to %c3 step %c1 : i8 {
      scf.for %j = %c0 to %i step %c1 : i8 {
      } loc("L")
    }
    tt.return
  })",
     3,
     {V::Refuted}},
    // Entered once per outer iteration; each invocation runs exactly once.
    {"inner_iv_plus_one_n1",
     R"(
  tt.func @f(%N: i4) {
    %c0 = arith.constant 0 : i4
    %c1 = arith.constant 1 : i4
    scf.for %i = %c0 to %N step %c1 : i4 {
      %ip1 = arith.addi %i, %c1 : i4
      scf.for %j = %i to %ip1 step %c1 : i4 {
      } loc("L")
    }
    tt.return
  })",
     1,
     {V::Satisfied}},
    {"inner_iv_plus_one_n2",
     R"(
  tt.func @f(%N: i4) {
    %c0 = arith.constant 0 : i4
    %c1 = arith.constant 1 : i4
    scf.for %i = %c0 to %N step %c1 : i4 {
      %ip1 = arith.addi %i, %c1 : i4
      scf.for %j = %i to %ip1 step %c1 : i4 {
      } loc("L")
    }
    tt.return
  })",
     2,
     {V::Refuted}},
    {"third_level_n2",
     R"(
  tt.func @f(%N: i4) {
    %c0 = arith.constant 0 : i4
    %c1 = arith.constant 1 : i4
    scf.for %i = %c0 to %N step %c1 : i4 {
      scf.for %j = %c0 to %N step %c1 : i4 {
        %jp1 = arith.addi %j, %c1 : i4
        scf.for %k = %j to %jp1 step %c1 : i4 {
        } loc("L")
      }
    }
    tt.return
  })",
     2,
     {V::Refuted}},
    {"unsigned_outer_n2",
     R"(
  tt.func @f() {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c2 = arith.constant 2 : i8
    scf.for unsigned %i = %c0 to %c2 step %c1 : i8 {
      scf.for %j = %c0 to %i step %c1 : i8 {
      } loc("L")
    }
    tt.return
  })",
     2,
     {V::Refuted}},
    // An assume just before the nested loop applies to that loop; the guard-
    // free Satisfied must hold in every launch the assume allows.
    {"assume_before_nested_n2",
     R"(
  tt.func @f(%N: i4, %K: i4) {
    %c0 = arith.constant 0 : i4
    %c1 = arith.constant 1 : i4
    %c2 = arith.constant 2 : i4
    scf.for %i = %c0 to %K step %c1 : i4 {
      %a = arith.cmpi sge, %N, %c2 : i4
      llvm.intr.assume %a : i1
      scf.for %j = %c0 to %N step %c1 : i4 {
      } loc("L")
    }
    tt.return
  })",
     2,
     {V::Satisfied}},
};

class TripCountOracleTest
    : public OracleFixture,
      public ::testing::WithParamInterface<TripOracleCase> {};

TEST_P(TripCountOracleTest, VerdictMatchesExhaustiveExecution) {
  using tt::intel::BoundProof;
  const TripOracleCase &c = GetParam();
  parse(c.ir);
  scf::ForOp loop = forNamed("L");
  BoundProof p = prover->tripCountAtLeast(loop, c.n);
  EXPECT_TRUE(llvm::is_contained(c.allowed, p.verdict))
      << c.name << ": " << toString(p);
  if (p.verdict == BoundProof::Unknown)
    return;
  Value guard = p.verdict == BoundProof::ConditionallySatisfied
                    ? materializeBeforeLoop(p, loop)
                    : Value();
  unsigned valid = 0, invocations = 0, guardedInvocations = 0;
  for (auto args : allArgumentValues(*module)) {
    auto trips = oracle.runTripCounts(module.get(), args, loop);
    if (!trips)
      continue; // UB, assume-violating, or outside the scf.for contract
    ++valid;
    invocations += trips->size();
    bool guardTrue = guard && oracle.evalGuard(guard, args);
    if (guardTrue)
      guardedInvocations += trips->size();
    for (uint64_t t : *trips) {
      switch (p.verdict) {
      case BoundProof::Satisfied:
        EXPECT_GE(t, static_cast<uint64_t>(c.n)) << c.name << args;
        break;
      case BoundProof::Refuted:
        EXPECT_LT(t, static_cast<uint64_t>(c.n)) << c.name << args;
        break;
      case BoundProof::ConditionallySatisfied:
        if (guardTrue)
          EXPECT_GE(t, static_cast<uint64_t>(c.n)) << c.name << args;
        break;
      case BoundProof::Unknown:
        break;
      }
    }
  }
  EXPECT_GT(valid, 0u) << c.name;
  // An "every invocation" claim over no invocation proves nothing.
  EXPECT_GT(invocations, 0u)
      << c.name << ": the queried loop was never entered";
  if (guard && !c.vacuous)
    EXPECT_GT(guardedInvocations, 0u)
        << c.name << ": no invocation ever ran under a true guard";
  RecordProperty(std::string(c.name) + "_valid", valid);
  RecordProperty(std::string(c.name) + "_invocations", invocations);
  RecordProperty(std::string(c.name) + "_guarded_invocations",
                 guardedInvocations);
}

static std::string
tripCaseName(const ::testing::TestParamInfo<TripOracleCase> &info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(TripCounts, TripCountOracleTest,
                         ::testing::ValuesIn(kTripCases), tripCaseName);

//===----------------------------------------------------------------------===//
// Guard arithmetic
//===----------------------------------------------------------------------===//

class SymbolicBoundsGuardTest : public OracleFixture {};

// Guard arithmetic near the i64 fit limit, independent of the prover.
// k = 2^55 takes the fast path (2^55*2^7 < 2^63); 2^56 is the first checked
// one. The guard must equal k*a >= t in exact arithmetic, and be false wherever
// k*a overflows i64. SymbolicBoundsGuardTest is the oracle fixture
// without the parameter.
TEST_F(SymbolicBoundsGuardTest, GuardMatchesExactArithmetic) {
  parse(R"(tt.func @f(%a: i8) { tt.return })");
  using namespace tt::intel;
  for (int64_t k :
       {int64_t(1) << 55, int64_t(1) << 56, int64_t(1) << 62, INT64_MIN})
    for (int64_t t : {int64_t(0), int64_t(1) << 60}) {
      BoundCondition c{
          AffineForm::symbol(prover->symbolFor(SymbolKind::KernelArg, arg(0)))
              .scale(k),
          BoundGoal::AtLeast, t};
      Operation *term = func().getBody().front().getTerminator();
      OpBuilder b(term);
      Value guard = materialize({c}, term, b);
      for (int64_t a = -128; a <= 127; ++a) {
        I128 exact = I128(k) * a;
        bool fits = exact >= INT64_MIN && exact <= INT64_MAX;
        EXPECT_EQ(oracle.evalGuard(guard, {a}), fits && exact >= t)
            << k << " " << t << " " << a;
      }
    }
}

//===----------------------------------------------------------------------===//
// Legacy guard implication
//===----------------------------------------------------------------------===//
//
// Coverage, not safety: wherever a legacy validator
// versions a loop, the guard of the prover's proof for the same mask must hold
// too. Per point, the legacy guard and the mask are evaluated at the family's
// width with wrapping, the new guard as `materialize` emits it, and legacy =>
// new is required except at
//
//   legacy-unsound  a mask element is false; the new guard must reject it
//   conservative    a sound point the new guard rejects because the prover
//                   will not reason through a wrapped value: (i) an unsigned
//                   predicate with a signed-negative side, (ii) an IR operand
//                   whose exact value exceeds its width
//
// Any other loss fails, and so does a new guard that holds over a false
// element. The canonical tutorial-03 K loop shape is legacy-recognized at
// i4/i8 and runs as real IR there; the scalar form's validator asserts i1
// operands, so it runs as real i1 IR, and its other widths model a build
// without assertions. No
// other family has recognized IR below i32 (tt.make_range is i32-only), so at
// i4/i8 they run on a model whose new guard is the i32 proof with its
// width-dependent constants substituted. Every family also runs at i32/i64 on
// boundary values, without a loop.

using Pred = arith::CmpIPredicate;

bool isUnsignedPred(Pred p) {
  return p == Pred::ult || p == Pred::ule || p == Pred::ugt || p == Pred::uge;
}

bool isLessThanPred(Pred p) {
  return p == Pred::slt || p == Pred::sle || p == Pred::ult || p == Pred::ule;
}

I128 sminOf(unsigned w) { return -(I128(1) << (w - 1)); }
I128 smaxOf(unsigned w) { return (I128(1) << (w - 1)) - 1; }
bool fitsSigned(I128 v, unsigned w) { return v >= sminOf(w) && v <= smaxOf(w); }

/// Floor division by a positive divisor.
I128 floorDiv(I128 a, I128 b) {
  I128 q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}

/// The indices of [0, n) at which `e0 + step*j`, wrapped to `w` bits, must be
/// evaluated to decide a comparison against a fixed value at every index: the
/// first and last, and both sides of each discontinuity of the predicate's
/// order, at INT_MIN(w) signed and at 0 unsigned. In between, the wrapped
/// value is monotone in j, so endpoints decide each stretch; the endpoints of
/// the whole range do not (`INT_MAX - 1 + [0..3] slt INT_MAX` is [T, F, T, T]).
SmallVector<I128, 8> discontinuitySamples(I128 e0, I128 step, I128 n,
                                          unsigned w, bool isUnsigned) {
  SmallVector<I128, 8> out;
  if (n <= 0)
    return out;
  out.push_back(0);
  out.push_back(n - 1);
  const I128 period = I128(1) << w;
  const I128 u0 = e0 - (isUnsigned ? I128(0) : sminOf(w));
  const I128 uLast = u0 + step * (n - 1);
  const I128 lo = std::min(floorDiv(u0, period), floorDiv(uLast, period));
  const I128 hi = std::max(floorDiv(u0, period), floorDiv(uLast, period));
  if (hi - lo > 4) {
    // No family wraps this often; refuse rather than under-sample.
    ADD_FAILURE() << "discontinuitySamples: " << static_cast<int64_t>(hi - lo)
                  << " wraps";
    return out;
  }
  for (I128 m = lo + 1; m <= hi; ++m) {
    // The first index past the edge m*period, rising or falling.
    I128 edge = m * period;
    I128 j =
        step > 0 ? -floorDiv(u0 - edge, step) : floorDiv(u0 - edge, -step) + 1;
    out.push_back(std::clamp<I128>(j - 1, 0, n - 1));
    out.push_back(std::clamp<I128>(j, 0, n - 1));
  }
  llvm::sort(out);
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Every value of a `w`-bit integer, as signed representatives.
std::vector<int64_t> allValues(unsigned w) {
  std::vector<int64_t> out;
  for (I128 v = sminOf(w); v <= smaxOf(w); ++v)
    out.push_back(static_cast<int64_t>(v));
  return out;
}

/// Boundary values of one argument: 0, +-1, +-(END-1), +-END,
/// INT_MIN, INT_MIN+END, INT_MAX-END and INT_MAX, plus INT_MIN+1 and INT_MAX-1,
/// which the legacy counterexamples need, and +-2*END, INT_MIN+(END-1) and
/// INT_MAX-(END-1): without them there is no multiple of END above END, so the
/// canonical implication, whose legacy-true K run from 2*END to
/// INT_MAX-(END-1), would be vacuous. END = 0 (no END) merges the sets for END
/// 2, 4 and 8.
std::vector<int64_t> boundaryValues(unsigned w, int64_t end) {
  std::vector<int64_t> out;
  for (int64_t e :
       end ? std::vector<int64_t>{end} : std::vector<int64_t>{2, 4, 8}) {
    const I128 mn = sminOf(w), mx = smaxOf(w);
    for (I128 v : {I128(0), I128(1), I128(-1), I128(e - 1), I128(1 - e),
                   I128(e), I128(-e), I128(2 * e), I128(-2 * e), mn, mn + 1,
                   mn + (e - 1), mn + e, mx - e, mx - (e - 1), mx - 1, mx})
      out.push_back(static_cast<int64_t>(v));
  }
  llvm::sort(out);
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Calls `fn` on every `nargs`-tuple (1 or 2) of `vals`.
void forEachTuple(const std::vector<int64_t> &vals, unsigned nargs,
                  llvm::function_ref<void(ArrayRef<int64_t>)> fn) {
  for (int64_t a : vals) {
    if (nargs == 1) {
      fn({a});
      continue;
    }
    for (int64_t b : vals) {
      int64_t t[2] = {a, b};
      fn(t);
    }
  }
}

//===----------------------------------------------------------------------===//
// Families and their model
//===----------------------------------------------------------------------===//

enum class LegacyFamily {
  Canonical,    // lane < K - k*END under cdiv(K, END):  K % END == 0 && K > END
  Scalar,       // N pred M:                             N pred M
  RangeLtSplat, // make_range(0, END) pred splat(N):     END-1 pred N
  SplatLtRange, // splat(N) pred make_range(0, END):     N pred 0
  AndRanges,    // (splat(M) pred' range) & (range pred splat(N)): both
  Boundary      // (splat(off) + ext(range)) pred c:  off + (END-1) pred c for
                // less-than predicates, narrow and unchecked, else off pred c
};

/// One `arith.cmpi` an invariant family is made of. The side that varies is
/// `base + lane`: base 0 for a bare make_range, the offset for the boundary
/// check, and N itself (a single lane) for the scalar form.
enum class Form { Scalar, LaneVsArg, ArgVsLane, OffsetLaneVsConst };

struct Conjunct {
  Form form;
  Pred pred;
  unsigned arg;      // N, M or the offset
  unsigned arg2 = 0; // the scalar form's M
};

/// andi's lower side, `M pred' lanes`, in the upper side's signedness.
Pred lowerOf(Pred upper) {
  return isUnsignedPred(upper) ? Pred::ule : Pred::sle;
}

SmallVector<Conjunct, 2> conjunctsOf(LegacyFamily f, Pred pred) {
  switch (f) {
  case LegacyFamily::Scalar:
    return {{Form::Scalar, pred, 0, 1}};
  case LegacyFamily::RangeLtSplat:
    return {{Form::LaneVsArg, pred, 0}};
  case LegacyFamily::SplatLtRange:
    return {{Form::ArgVsLane, pred, 0}};
  case LegacyFamily::AndRanges:
    return {{Form::ArgVsLane, lowerOf(pred), 0}, {Form::LaneVsArg, pred, 1}};
  case LegacyFamily::Boundary:
    return {{Form::OffsetLaneVsConst, pred, 0}};
  case LegacyFamily::Canonical:
    break;
  }
  llvm_unreachable("the canonical family has no invariant conjunct");
}

unsigned numArgs(LegacyFamily f) {
  return f == LegacyFamily::Scalar || f == LegacyFamily::AndRanges ? 2 : 1;
}

/// END values per width; i4 cannot hold the lanes of END = 8. The scalar form
/// has no END, written 0.
SmallVector<int64_t, 3> endsFor(LegacyFamily f, unsigned w) {
  if (f == LegacyFamily::Scalar)
    return {0};
  if (w == 4)
    return {2, 4};
  return {2, 4, 8};
}

struct ConjunctEval {
  SmallVector<bool, 8> lanes; // every lane: the exact mask
  bool sampledAllTrue = true; // only the lanes discontinuitySamples picks
  bool legacy = false;
  bool kindI = false;  // unsigned predicate, a signed-negative side
  bool kindII = false; // base + lane exceeds the width
};

/// One conjunct at one point, with `a` at width `w` and `c` the boundary
/// family's bound, in APInt with wrapping as the hardware computes it.
ConjunctEval evalConjunct(const Conjunct &cj, ArrayRef<APInt> a, unsigned w,
                          int64_t end, int64_t c) {
  ConjunctEval r;
  const bool scalar = cj.form == Form::Scalar;
  const int64_t lanes = scalar ? 1 : end;
  const bool offset = cj.form == Form::OffsetLaneVsConst;
  const APInt base = scalar || offset ? a[cj.arg] : APInt(w, 0);
  const APInt other = scalar   ? a[cj.arg2]
                      : offset ? wrapToWidth(c, w)
                               : a[cj.arg];
  auto at = [&](I128 l) {
    APInt v = base + wrapToWidth(l, w);
    return cj.form == Form::ArgVsLane ? applyCmp(cj.pred, other, v)
                                      : applyCmp(cj.pred, v, other);
  };
  for (int64_t l = 0; l < lanes; ++l)
    r.lanes.push_back(at(l));
  for (I128 l : discontinuitySamples(base.getSExtValue(), 1, lanes, w,
                                     isUnsignedPred(cj.pred)))
    r.sampledAllTrue &= at(l);

  // The guard InvariantMaskValidator::getVersioningCond builds for the form.
  switch (cj.form) {
  case Form::Scalar:
    r.legacy = applyCmp(cj.pred, a[cj.arg], a[cj.arg2]);
    break;
  case Form::LaneVsArg:
    r.legacy = applyCmp(cj.pred, wrapToWidth(end - 1, w), a[cj.arg]);
    break;
  case Form::ArgVsLane:
    r.legacy = applyCmp(cj.pred, a[cj.arg], APInt(w, 0));
    break;
  case Form::OffsetLaneVsConst: {
    int64_t adjust = isLessThanPred(cj.pred) ? end - 1 : 0;
    r.legacy = applyCmp(cj.pred, a[cj.arg] + wrapToWidth(adjust, w),
                        wrapToWidth(c, w));
    break;
  }
  }

  if (isUnsignedPred(cj.pred)) {
    r.kindI = other.isNegative();
    for (int64_t l = 0; l < lanes; ++l)
      r.kindI |= (base + wrapToWidth(l, w)).isNegative();
  }
  r.kindII = !fitsSigned(I128(base.getSExtValue()) + (lanes - 1), w);
  return r;
}

struct CanonicalEval {
  bool allTrue = true, anyTrue = false;
  I128 trips = 0;
  bool exceedsWidth = false; // kind (ii): some IR operand exceeds the width
};

/// The tutorial-03 K loop mask at width `w`: `for k in [0, (K + END-1) / END)`,
/// lanes [0, END) slt K - k*END, all in `w`-bit wrapping arithmetic, compared
/// at i32 (a narrow rem is sign-extended) or i64. The lanes enter unshifted, so
/// every lane is evaluated; the iterations are all run, or only those
/// discontinuitySamples picks, which is what the i32/i64 points need.
CanonicalEval canonicalMask(int64_t k, unsigned w, int64_t end,
                            bool allIterations) {
  CanonicalEval r;
  const APInt K = wrapToWidth(k, w), E = wrapToWidth(end, w);
  const APInt q = (K + wrapToWidth(end - 1, w)).sdiv(E);
  r.exceedsWidth = !fitsSigned(I128(k) + (end - 1), w);
  r.trips = std::max<I128>(0, q.getSExtValue());
  const unsigned cw = std::max(w, 32u);
  SmallVector<I128, 8> iterations;
  if (allIterations)
    for (I128 j = 0; j < r.trips; ++j)
      iterations.push_back(j);
  else
    iterations = discontinuitySamples(k, -end, r.trips, w, false);
  for (I128 j : iterations) {
    APInt rem = (K - wrapToWidth(j, w) * E).sextOrTrunc(cw);
    r.exceedsWidth |=
        !fitsSigned(j * end, w) || !fitsSigned(I128(k) - j * end, w);
    for (int64_t l = 0; l < end; ++l) {
      bool b = APInt(cw, l).slt(rem);
      r.allTrue &= b;
      r.anyTrue |= b;
    }
  }
  return r;
}

/// CanonicalMaskValidator::getVersioningCond: `N % END == 0 && N > END`.
bool canonicalLegacy(int64_t k, unsigned w, int64_t end) {
  const APInt K = wrapToWidth(k, w), E = wrapToWidth(end, w);
  return K.srem(E).isZero() && K.sgt(E);
}

//===----------------------------------------------------------------------===//
// Fixture IR
//===----------------------------------------------------------------------===//

std::string intTy(unsigned w) { return "i" + std::to_string(w); }

std::string tensorTy(int64_t n, unsigned w) {
  return "tensor<" + std::to_string(n) + "x" + intTy(w) + ">";
}

/// The tutorial-03 K loop shape at width `w`, with an `arith.select` `use`
/// consuming the mask as the masked load would: the symbolic driver proves the
/// mask there.
std::string canonicalIR(unsigned w, int64_t end) {
  const std::string iw = intTy(w), e = std::to_string(end);
  const unsigned cw = std::max(w, 32u);
  const std::string t32 = tensorTy(end, 32), tc = tensorTy(end, cw);
  std::string lane = "%lane", rem = "%rem";
  std::string ir = "tt.func @cdiv_k_loop(%K: " + iw + ") {\n";
  ir += "  %c0 = arith.constant 0 : " + iw + "\n";
  ir += "  %c1 = arith.constant 1 : " + iw + "\n";
  ir += "  %cm = arith.constant " + std::to_string(end - 1) + " : " + iw + "\n";
  ir += "  %ce = arith.constant " + e + " : " + iw + "\n";
  ir += "  %num = arith.addi %K, %cm : " + iw + "\n";
  ir += "  %q = arith.divsi %num, %ce : " + iw + "\n";
  ir += "  %lane = tt.make_range {start = 0 : i32, end = " + e +
        " : i32} : " + t32 + "\n";
  if (cw == 64) {
    ir += "  %lane64 = arith.extsi %lane : " + t32 + " to " + tc + "\n";
    lane = "%lane64";
  }
  ir += "  scf.for %k = %c0 to %q step %c1 : " + iw + " {\n";
  ir += "    %ke = arith.muli %k, %ce : " + iw + "\n";
  ir += "    %rem = arith.subi %K, %ke : " + iw + "\n";
  if (w < 32) {
    ir += "    %remc = arith.extsi %rem : " + iw + " to i32\n";
    rem = "%remc";
  }
  ir += "    %rs = tt.splat " + rem + " : " + intTy(cw) + " -> " + tc + "\n";
  ir += "    %mask = arith.cmpi slt, " + lane + ", %rs : " + tc +
        " loc(\"mask\")\n";
  ir += "    %use = arith.select %mask, %lane, %lane : " + tensorTy(end, 1) +
        ", " + t32 + " loc(\"use\")\n";
  return ir + "  }\n  tt.return\n}\n";
}

/// An invariant family at width `w`: each mask (one per bound `bounds[i]` for
/// the boundary check, else one) is computed before a one-iteration loop and
/// consumed inside it by an `arith.select` `u<i>`. At i64 the lanes are
/// sign-extended, as RewriteTensorDescriptorToPointer emits them.
std::string invariantIR(LegacyFamily f, Pred pred, unsigned w, int64_t end,
                        ArrayRef<int64_t> bounds) {
  const int64_t n = f == LegacyFamily::Scalar ? 4 : end;
  const std::string iw = intTy(w), t = tensorTy(n, w), t32 = tensorTy(n, 32),
                    t1 = tensorTy(n, 1),
                    p = arith::stringifyCmpIPredicate(pred).str();
  std::string lanes = "%r";
  std::string ir = "tt.func @f(%a0: " + iw +
                   (numArgs(f) == 2 ? ", %a1: " + iw : std::string()) + ") {\n";
  ir += "  %c0 = arith.constant 0 : i32\n  %c1 = arith.constant 1 : i32\n";
  ir += "  %r = tt.make_range {start = 0 : i32, end = " + std::to_string(n) +
        " : i32} : " + t32 + "\n";
  if (w == 64 && f != LegacyFamily::Scalar) {
    ir += "  %rw = arith.extsi %r : " + t32 + " to " + t + "\n";
    lanes = "%rw";
  }
  unsigned masks = 1;
  switch (f) {
  case LegacyFamily::Scalar:
    ir += "  %cmp = arith.cmpi " + p + ", %a0, %a1 : " + iw + "\n";
    ir += "  %m0 = tt.splat %cmp : i1 -> " + t1 + "\n";
    break;
  case LegacyFamily::RangeLtSplat:
  case LegacyFamily::SplatLtRange: {
    bool laneFirst = f == LegacyFamily::RangeLtSplat;
    ir += "  %s0 = tt.splat %a0 : " + iw + " -> " + t + "\n";
    ir += "  %m0 = arith.cmpi " + p + ", " + (laneFirst ? lanes : "%s0") +
          ", " + (laneFirst ? "%s0" : lanes) + " : " + t + "\n";
    break;
  }
  case LegacyFamily::AndRanges:
    ir += "  %s0 = tt.splat %a0 : " + iw + " -> " + t + "\n";
    ir += "  %s1 = tt.splat %a1 : " + iw + " -> " + t + "\n";
    ir += "  %lo = arith.cmpi " +
          arith::stringifyCmpIPredicate(lowerOf(pred)).str() + ", %s0, " +
          lanes + " : " + t + " loc(\"lo\")\n";
    ir += "  %hi = arith.cmpi " + p + ", " + lanes + ", %s1 : " + t +
          " loc(\"hi\")\n";
    ir += "  %m0 = arith.andi %lo, %hi : " + t1 + "\n";
    break;
  case LegacyFamily::Boundary:
    ir += "  %s0 = tt.splat %a0 : " + iw + " -> " + t + "\n";
    ir += "  %idx = arith.addi %s0, " + lanes + " : " + t + "\n";
    masks = bounds.size();
    for (unsigned i = 0; i < masks; ++i) {
      std::string s = std::to_string(i);
      ir += "  %b" + s + " = arith.constant dense<" +
            std::to_string(bounds[i]) + "> : " + t + "\n";
      ir += "  %m" + s + " = arith.cmpi " + p + ", %idx, %b" + s + " : " + t +
            "\n";
    }
    break;
  case LegacyFamily::Canonical:
    llvm_unreachable("canonicalIR builds the canonical family");
  }
  ir += "  scf.for %i = %c0 to %c1 step %c1 : i32 {\n";
  for (unsigned i = 0; i < masks; ++i) {
    std::string s = std::to_string(i);
    ir += "    %u" + s + " = arith.select %m" + s + ", %r, %r : " + t1 + ", " +
          t32 + " loc(\"u" + s + "\")\n";
  }
  return ir + "  }\n  tt.return\n}\n";
}

/// Only arguments of width `w`: the narrow model has no IR of its own, so its
/// guards are materialized here, onto symbols of the right width.
std::string shellIR(unsigned w, unsigned nargs) {
  return "tt.func @g(%a0: " + intTy(w) +
         (nargs == 2 ? ", %a1: " + intTy(w) : std::string()) +
         ") {\n  tt.return\n}\n";
}

//===----------------------------------------------------------------------===//
// Proofs, guards and the width substitution
//===----------------------------------------------------------------------===//

/// A parsed module with its own analyses, so that the i32 and i64 modules and
/// a narrow shell of one family can be alive at the same time.
struct AnalyzedModule {
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<DataFlowSolver> solver;
  std::unique_ptr<tt::intel::SymbolicBoundsProver> prover;
  llvm::StringMap<Operation *> named; // loc("<name>")

  Operation *op(StringRef name) const {
    Operation *found = named.lookup(name);
    EXPECT_TRUE(found) << "no op named '" << name.str() << "'";
    return found;
  }
  tt::FuncOp func() {
    tt::FuncOp f;
    module->walk([&](tt::FuncOp op) { f = op; });
    return f;
  }
  Value arg(unsigned i) { return func().getArgument(i); }
};

std::unique_ptr<AnalyzedModule> analyzeModule(MLIRContext &ctx,
                                              const std::string &ir) {
  auto am = std::make_unique<AnalyzedModule>();
  am->module = parseSourceString<ModuleOp>(ir, &ctx);
  if (!am->module) {
    ADD_FAILURE() << "failed to parse:\n" << ir;
    return nullptr;
  }
  ModuleOp mod = am->module.get();
  am->domInfo = std::make_unique<DominanceInfo>(mod);
  am->solver = createDataFlowSolver();
  am->solver->load<tt::intel::IntegerRangeAnalysis>(mod, *am->domInfo);
  if (failed(am->solver->initializeAndRun(mod))) {
    ADD_FAILURE() << "range analysis failed on:\n" << ir;
    return nullptr;
  }
  am->prover = std::make_unique<tt::intel::SymbolicBoundsProver>(
      *am->solver, *am->domInfo, mod);
  mod.walk([&](Operation *op) {
    if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc()))
      am->named[nameLoc.getName().getValue()] = op;
  });
  return am;
}

/// The proof RemoveMasks' symbolic driver obtains for `mask`, consumed by
/// `use`: proveTrue at `use`, inside its loop.
tt::intel::BoundProof driverProof(AnalyzedModule &am, Operation *use,
                                  Value mask) {
  return am.prover->proveTrue(mask, {use, use->getParentOfType<scf::ForOp>()});
}

struct NewGuard {
  tt::intel::BoundProof::Verdict verdict = tt::intel::BoundProof::Unknown;
  Value guard; // ConditionallySatisfied only
  std::string text;
};

/// Materializes `p` before `before`, as the driver does before the loop.
NewGuard materializeAt(const tt::intel::BoundProof &p, Operation *before) {
  NewGuard g{p.verdict, Value(), toString(p)};
  if (p.verdict == tt::intel::BoundProof::ConditionallySatisfied) {
    OpBuilder b(before);
    g.guard = tt::intel::materialize(p.conditions, before, b);
  }
  return g;
}

/// Unknown versions nothing and Refuted never unmasks: neither is a fast path.
bool holds(Oracle &oracle, const NewGuard &g, ArrayRef<int64_t> args) {
  switch (g.verdict) {
  case tt::intel::BoundProof::Satisfied:
    return true;
  case tt::intel::BoundProof::ConditionallySatisfied:
    return oracle.evalGuard(g.guard, args);
  default:
    return false;
  }
}

/// An i32 proof's condition with the width taken out: the wrap
/// guards AtMost(X, INT_MAX(32) - (END-1)) and AtLeast(X, INT_MIN(32)) take
/// the width as a parameter; DivisibleBy, sign and residual conditions keep
/// their constants. Subjects are over kernel arguments, by index.
struct CondTemplate {
  enum Bound { Literal, MaxLessEnd, Min };
  SmallVector<std::pair<unsigned, int64_t>, 2> terms;
  int64_t c0;
  tt::intel::BoundGoal goal;
  Bound bound;
  int64_t c;
  tt::intel::ConditionKind kind;
};

/// The narrow-model families' residual constants derive from i8 constants and
/// END; anything larger came from a width.
constexpr int64_t kResidualLimit = int64_t(1) << 16;

bool isResidualConstant(int64_t v) {
  return v >= -kResidualLimit && v <= kResidualLimit;
}

/// Returns nullopt, with the reason, for a condition the mapping cannot carry:
/// a subject over anything but kernel arguments, or a constant outside the
/// set above. Such a family is left out of the narrow model.
std::optional<SmallVector<CondTemplate, 4>>
templateOf(ArrayRef<tt::intel::BoundCondition> conds, int64_t end,
           std::string &why) {
  using tt::intel::BoundGoal;
  SmallVector<CondTemplate, 4> out;
  for (const tt::intel::BoundCondition &cond : conds) {
    CondTemplate t{{},        cond.expr.constant(),
                   cond.goal, CondTemplate::Literal,
                   cond.c,    cond.kind};
    for (auto &[sym, k] : cond.expr.terms()) {
      auto arg = dyn_cast_if_present<BlockArgument>(sym.value());
      if (sym.kind() != tt::intel::SymbolKind::KernelArg || !arg) {
        why = "'" + toString(cond) + "' is not over kernel arguments";
        return std::nullopt;
      }
      t.terms.push_back({arg.getArgNumber(), k});
    }
    bool ordered =
        cond.goal == BoundGoal::AtMost || cond.goal == BoundGoal::AtLeast;
    if (cond.goal == BoundGoal::AtMost && cond.c == INT32_MAX - (end - 1))
      t.bound = CondTemplate::MaxLessEnd;
    else if (cond.goal == BoundGoal::AtLeast && cond.c == INT32_MIN)
      t.bound = CondTemplate::Min;
    else if ((ordered && !isResidualConstant(cond.c)) ||
             !isResidualConstant(t.c0)) {
      why = "'" + toString(cond) +
            "' has a width-dependent constant that "
            "is neither wrap form";
      return std::nullopt;
    }
    out.push_back(std::move(t));
  }
  return out;
}

/// The template at width `w`, over the arguments of `am`'s function.
SmallVector<tt::intel::BoundCondition, 4> instantiate(ArrayRef<CondTemplate> ts,
                                                      unsigned w, int64_t end,
                                                      AnalyzedModule &am) {
  using namespace tt::intel;
  SmallVector<BoundCondition, 4> out;
  for (const CondTemplate &t : ts) {
    AffineForm e = AffineForm::constant(t.c0);
    for (auto [idx, k] : t.terms)
      e = e.add(AffineForm::symbol(
                    am.prover->symbolFor(SymbolKind::KernelArg, am.arg(idx)))
                    .scale(k));
    int64_t c = t.bound == CondTemplate::MaxLessEnd
                    ? static_cast<int64_t>(smaxOf(w)) - (end - 1)
                : t.bound == CondTemplate::Min ? static_cast<int64_t>(sminOf(w))
                                               : t.c;
    out.push_back({e, t.goal, c, t.kind});
  }
  return out;
}

/// Structural equality, kinds included, in order: materialize emits in order.
bool sameConditions(ArrayRef<tt::intel::BoundCondition> a,
                    ArrayRef<tt::intel::BoundCondition> b) {
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip(a, b))
    if (!(x == y) || x.kind != y.kind)
      return false;
  return true;
}

std::string render(ArrayRef<tt::intel::BoundCondition> cs) {
  std::string out = "{";
  for (auto [i, c] : llvm::enumerate(cs))
    out += (i ? "; " : "") + toString(c);
  return out + "}";
}

//===----------------------------------------------------------------------===//
// Classification and the report
//===----------------------------------------------------------------------===//

enum class PointClass {
  LegacyFalse,
  Covered,
  LegacyUnsound,
  LossUnsigned,    // conservative (i)
  LossWrap,        // conservative (ii)
  LossUnclassified // a coverage regression
};

/// Per conjunct of a lost point: whether its own new guard holds, and which
/// conservative kind its operands show.
struct ConjunctLoss {
  bool newOk, kindI, kindII;
};

/// A loss is explained only if some conjunct's guard rejects the point and
/// every rejecting conjunct shows kind (i) or (ii); a conjunction that loses
/// what both its sides keep is not.
PointClass classify(bool legacy, bool allTrue, bool newOk,
                    ArrayRef<ConjunctLoss> conjuncts) {
  if (!legacy)
    return PointClass::LegacyFalse;
  if (!allTrue)
    return PointClass::LegacyUnsound;
  if (newOk)
    return PointClass::Covered;
  bool rejected = false, explained = true, unsignedKind = false;
  for (const ConjunctLoss &cl : conjuncts) {
    if (cl.newOk)
      continue;
    rejected = true;
    unsignedKind |= cl.kindI;
    explained &= cl.kindI || cl.kindII;
  }
  if (!rejected || !explained)
    return PointClass::LossUnclassified;
  return unsignedKind ? PointClass::LossUnsigned : PointClass::LossWrap;
}

/// The points of one class in one group: the count, the range each argument
/// and the bound span, and the first few points.
struct ClassLog {
  uint64_t count = 0;
  SmallVector<std::pair<int64_t, int64_t>, 2> argRange;
  std::optional<std::pair<int64_t, int64_t>> boundRange;
  SmallVector<std::string, 4> examples;

  void add(ArrayRef<int64_t> args, std::optional<int64_t> bound,
           llvm::function_ref<std::string()> detail) {
    if (count++ == 0) {
      for (int64_t v : args)
        argRange.push_back({v, v});
      if (bound)
        boundRange = {*bound, *bound};
    }
    for (unsigned i = 0; i < args.size(); ++i) {
      argRange[i].first = std::min(argRange[i].first, args[i]);
      argRange[i].second = std::max(argRange[i].second, args[i]);
    }
    if (bound && boundRange) {
      boundRange->first = std::min(boundRange->first, *bound);
      boundRange->second = std::max(boundRange->second, *bound);
    }
    if (examples.size() < 3)
      examples.push_back(detail());
  }

  std::string str() const {
    std::string s = std::to_string(count) + " at";
    for (auto [i, r] : llvm::enumerate(argRange))
      s += " arg" + std::to_string(i) + " in [" + std::to_string(r.first) +
           ", " + std::to_string(r.second) + "]";
    if (boundRange)
      s += " c in [" + std::to_string(boundRange->first) + ", " +
           std::to_string(boundRange->second) + "]";
    for (const std::string &e : examples)
      s += "; " + e;
    return s;
  }
};

/// One (width, END) of one case.
struct Group {
  uint64_t points = 0, legacyTrue = 0, covered = 0, newTrue = 0;
  ClassLog legacyUnsound, lossUnsigned, lossWrap, lossUnclassified;
  ClassLog unsound;  // the new guard holds over a false element
  ClassLog mismatch; // the sampler or the mask model disagrees with execution
  unsigned excluded = 0;
  std::string exclusion;
};

class ImplicationLog {
public:
  explicit ImplicationLog(std::string name) : name(std::move(name)) {}

  void record(unsigned w, int64_t end, std::optional<int64_t> bound,
              ArrayRef<int64_t> args, PointClass cls, bool newOk, bool unsound,
              StringRef newText) {
    Group &g = groups[{w, end}];
    ++g.points;
    g.newTrue += newOk;
    auto detail = [&] {
      return pointText(args, bound) + " new=" + newText.str();
    };
    if (unsound)
      g.unsound.add(args, bound, detail);
    if (cls != PointClass::LegacyFalse)
      ++g.legacyTrue;
    switch (cls) {
    case PointClass::LegacyFalse:
      break;
    case PointClass::Covered:
      ++g.covered;
      break;
    case PointClass::LegacyUnsound:
      g.legacyUnsound.add(args, bound, detail);
      break;
    case PointClass::LossUnsigned:
      g.lossUnsigned.add(args, bound, detail);
      break;
    case PointClass::LossWrap:
      g.lossWrap.add(args, bound, detail);
      break;
    case PointClass::LossUnclassified:
      g.lossUnclassified.add(args, bound, detail);
      break;
    }
  }

  void mismatch(unsigned w, int64_t end, std::optional<int64_t> bound,
                ArrayRef<int64_t> args, StringRef what) {
    groups[{w, end}].mismatch.add(
        args, bound, [&] { return pointText(args, bound) + " " + what.str(); });
  }

  void exclude(unsigned w, int64_t end, const std::string &why) {
    Group &g = groups[{w, end}];
    if (g.excluded++ == 0)
      g.exclusion = why;
  }

  /// Prints the per-group table, records the totals, and fails on an
  /// unclassified loss, an unsound new guard, or a model mismatch.
  ///
  /// `knownCoverageGap`: the cases that set it still report their unclassified
  /// losses but are not failed by them; the unsound and mismatch checks stay
  /// fatal. Each loss is a sound but conservative answer outside the unsigned
  /// and IR-wrap exception classes:
  ///  - `scalar_slt`/`sle` at i64: the residual guard `-a + b >= k` is built
  ///    in overflow-checked arithmetic and is false whenever `-a` or the sum
  ///    leaves i64 (a == INT64_MIN, or b - a > INT64_MAX), though a < b holds.
  ///  - `scalar_slt`/`sle` at i1: the one-term term-sign guard (`b > 0`) and
  ///    the two-term residual guard tie in `pickWidest`, which keeps the
  ///    earlier, narrower one.
  ///  - `boundary_slt`/`sle`/`sge`/`sgt` at i64: the verdict is Unknown at the
  ///    sampled bounds next to INT64_MIN.
  void finish(bool legacyVacuous, bool knownCoverageGap = false) {
    uint64_t legacyTrue = 0, sound = 0, unsoundLegacy = 0, lossI = 0,
             lossII = 0, unclassified = 0;
    for (auto &[key, g] : groups) {
      std::string where =
          name + " w=" + std::to_string(key.first) +
          (key.second ? " END=" + std::to_string(key.second) : std::string());
      std::cout << "[ implication ] " << where << ": points=" << g.points
                << " legacy=" << g.legacyTrue << " covered=" << g.covered
                << " legacy-unsound=" << g.legacyUnsound.count
                << " loss(i)=" << g.lossUnsigned.count
                << " loss(ii)=" << g.lossWrap.count
                << " unclassified=" << g.lossUnclassified.count
                << " new-true=" << g.newTrue;
      if (g.excluded)
        std::cout << " excluded=" << g.excluded << " (" << g.exclusion << ")";
      std::cout << "\n";
      for (auto [label, entry] : {std::pair<const char *, const ClassLog *>{
                                      "legacy-unsound", &g.legacyUnsound},
                                  {"loss(i)", &g.lossUnsigned},
                                  {"loss(ii)", &g.lossWrap},
                                  {"unclassified", &g.lossUnclassified}})
        if (entry->count)
          std::cout << "      " << label << ": " << entry->str() << "\n";
      // Not silently dropped either way: the count and examples are already
      // printed above and counted into the RecordProperty totals below; this
      // is only whether a nonzero count is fatal for this named case.
      if (knownCoverageGap) {
        if (g.lossUnclassified.count)
          std::cout << "      [documented, not fatal: known coverage gap, see "
                       "finish()'s doc comment]\n";
      } else {
        EXPECT_EQ(g.lossUnclassified.count, 0u)
            << where << ": the new guard rejects sound legacy-guarded points "
            << "outside both exception classes: " << g.lossUnclassified.str();
      }
      EXPECT_EQ(g.unsound.count, 0u)
          << where << ": the new guard holds where a mask element is false: "
          << g.unsound.str();
      EXPECT_EQ(g.mismatch.count, 0u)
          << where
          << ": the model disagrees with execution: " << g.mismatch.str();
      legacyTrue += g.legacyTrue;
      sound += g.legacyTrue - g.legacyUnsound.count;
      unsoundLegacy += g.legacyUnsound.count;
      lossI += g.lossUnsigned.count;
      lossII += g.lossWrap.count;
      unclassified += g.lossUnclassified.count;
    }
    ::testing::Test::RecordProperty(name + "_legacy_unsound", unsoundLegacy);
    ::testing::Test::RecordProperty(name + "_loss_unsigned", lossI);
    ::testing::Test::RecordProperty(name + "_loss_wrap", lossII);
    ::testing::Test::RecordProperty(name + "_unclassified", unclassified);
    if (legacyVacuous)
      EXPECT_EQ(legacyTrue, 0u) << name << ": marked vacuous, yet the legacy "
                                << "guard holds somewhere";
    else
      EXPECT_GT(sound, 0u) << name << ": the legacy guard never soundly "
                           << "holds, so the implication was never tested";
  }

private:
  static std::string pointText(ArrayRef<int64_t> args,
                               std::optional<int64_t> bound) {
    std::string s = "(";
    for (auto [i, v] : llvm::enumerate(args))
      s += (i ? ", " : "") + std::to_string(v);
    s += ")";
    if (bound)
      s += " c=" + std::to_string(*bound);
    return s;
  }

  std::string name;
  std::map<std::pair<unsigned, int64_t>, Group> groups;
};

//===----------------------------------------------------------------------===//
// The implication test
//===----------------------------------------------------------------------===//

struct ImplicationCase {
  const char *name;
  LegacyFamily family;
  Pred pred; // the upper side's, for andi
  bool legacyVacuous = false;
  // Sound but conservative losses that do not fail the case; see
  // ImplicationLog::finish's doc comment.
  bool knownCoverageGap = false;
};

// gtest prints a failing parameter; without this it dumps the struct's bytes.
void PrintTo(const ImplicationCase &c, std::ostream *os) { *os << c.name; }

static const ImplicationCase kImplicationCases[] = {
    {"canonical", LegacyFamily::Canonical, Pred::slt},
    {"scalar_slt", LegacyFamily::Scalar, Pred::slt, false, true},
    {"scalar_sle", LegacyFamily::Scalar, Pred::sle, false, true},
    {"scalar_ult", LegacyFamily::Scalar, Pred::ult},
    {"scalar_ule", LegacyFamily::Scalar, Pred::ule},
    {"range_lt_splat_slt", LegacyFamily::RangeLtSplat, Pred::slt},
    {"range_lt_splat_sle", LegacyFamily::RangeLtSplat, Pred::sle},
    {"range_lt_splat_ult", LegacyFamily::RangeLtSplat, Pred::ult},
    {"range_lt_splat_ule", LegacyFamily::RangeLtSplat, Pred::ule},
    {"splat_lt_range_slt", LegacyFamily::SplatLtRange, Pred::slt},
    {"splat_lt_range_sle", LegacyFamily::SplatLtRange, Pred::sle},
    // N ult 0 never holds.
    {"splat_lt_range_ult", LegacyFamily::SplatLtRange, Pred::ult, true},
    {"splat_lt_range_ule", LegacyFamily::SplatLtRange, Pred::ule},
    {"andi_signed", LegacyFamily::AndRanges, Pred::slt},
    {"andi_unsigned", LegacyFamily::AndRanges, Pred::ult},
    {"boundary_slt", LegacyFamily::Boundary, Pred::slt, false, true},
    {"boundary_sle", LegacyFamily::Boundary, Pred::sle, false, true},
    {"boundary_ult", LegacyFamily::Boundary, Pred::ult},
    {"boundary_ule", LegacyFamily::Boundary, Pred::ule},
    {"boundary_sge", LegacyFamily::Boundary, Pred::sge, false, true},
    {"boundary_sgt", LegacyFamily::Boundary, Pred::sgt, false, true},
    {"boundary_uge", LegacyFamily::Boundary, Pred::uge},
    {"boundary_ugt", LegacyFamily::Boundary, Pred::ugt},
};

/// One mask of a module: the driver's proof and, for andi, each side's own
/// proof, which attributes a loss to the side that rejects it.
struct MaskInstance {
  Operation *loop = nullptr;
  tt::intel::BoundProof proof;
  SmallVector<tt::intel::BoundProof, 2> sides;
  NewGuard guard;
  SmallVector<NewGuard, 2> sideGuards;
};

class LegacyImplicationTest
    : public OracleFixture,
      public ::testing::WithParamInterface<ImplicationCase> {
protected:
  /// Every proof of a module is taken before any guard is materialized, as
  /// the driver's read-only analysis phase does.
  std::vector<MaskInstance> proveAll(AnalyzedModule &am, unsigned masks,
                                     bool withSides) {
    std::vector<MaskInstance> out(masks);
    for (unsigned i = 0; i < masks; ++i) {
      Operation *use = am.op("u" + std::to_string(i));
      out[i].loop = use->getParentOfType<scf::ForOp>();
      out[i].proof = driverProof(am, use, use->getOperand(0));
      if (withSides)
        for (StringRef side : {"lo", "hi"})
          out[i].sides.push_back(
              driverProof(am, use, am.op(side)->getResult(0)));
    }
    for (MaskInstance &mi : out) {
      mi.guard = materializeAt(mi.proof, mi.loop);
      for (const tt::intel::BoundProof &p : mi.sides)
        mi.sideGuards.push_back(materializeAt(p, mi.loop));
    }
    return out;
  }

  void checkInvariantPoint(ImplicationLog &log, ArrayRef<Conjunct> cjs,
                           unsigned w, int64_t end,
                           std::optional<int64_t> bound, ArrayRef<int64_t> args,
                           const NewGuard &guard,
                           ArrayRef<NewGuard> sideGuards) {
    SmallVector<APInt, 2> a;
    for (int64_t v : args)
      a.push_back(wrapToWidth(v, w));
    SmallVector<ConjunctEval, 2> ev;
    for (const Conjunct &cj : cjs)
      ev.push_back(evalConjunct(cj, a, w, end, bound.value_or(0)));
    bool allTrue = true, anyTrue = false, sampled = true, legacy = true;
    for (unsigned l = 0; l < ev.front().lanes.size(); ++l) {
      bool element = true;
      for (const ConjunctEval &e : ev)
        element &= e.lanes[l];
      allTrue &= element;
      anyTrue |= element;
    }
    for (const ConjunctEval &e : ev) {
      sampled &= e.sampledAllTrue;
      legacy &= e.legacy;
    }
    if (sampled != allTrue)
      log.mismatch(w, end, bound, args, "lane sampler");

    bool newOk = holds(oracle, guard, args);
    bool unsound = (newOk && !allTrue) ||
                   (guard.verdict == tt::intel::BoundProof::Refuted && anyTrue);
    SmallVector<ConjunctLoss, 2> losses;
    if (legacy && allTrue && !newOk)
      for (auto [j, e] : llvm::enumerate(ev))
        losses.push_back(
            {sideGuards.empty() ? newOk : holds(oracle, sideGuards[j], args),
             e.kindI, e.kindII});
    log.record(w, end, bound, args, classify(legacy, allTrue, newOk, losses),
               newOk, unsound, guard.text);
  }

  void checkCanonical(ImplicationLog &log);
  void checkInvariant(const ImplicationCase &c, ImplicationLog &log);
};

void LegacyImplicationTest::checkCanonical(ImplicationLog &log) {
  using tt::intel::BoundProof;
  std::map<int64_t, SmallVector<CondTemplate, 4>> t32; // by END
  // i32 first: its proof is the template the other widths are checked against.
  for (unsigned w : {32u, 4u, 8u, 64u}) {
    for (int64_t end : endsFor(LegacyFamily::Canonical, w)) {
      std::unique_ptr<AnalyzedModule> am =
          analyzeModule(ctx, canonicalIR(w, end));
      ASSERT_TRUE(am);
      Operation *use = am->op("use");
      ASSERT_TRUE(use);
      BoundProof p = driverProof(*am, use, use->getOperand(0));
      std::string where =
          "canonical w=" + std::to_string(w) + " END=" + std::to_string(end);

      // This family alone has recognized IR at every width, so its own
      // proofs are the ground truth for the width substitution.
      if (w == 32) {
        std::string why;
        auto t = templateOf(p.conditions, end, why);
        ASSERT_TRUE(t) << where << ": " << why;
        EXPECT_TRUE(sameConditions(instantiate(*t, 32, end, *am), p.conditions))
            << where << ": the substitution does not reproduce " << toString(p);
        t32[end] = *t;
      } else if (p.verdict == BoundProof::ConditionallySatisfied) {
        auto subst = instantiate(t32[end], w, end, *am);
        EXPECT_TRUE(sameConditions(subst, p.conditions))
            << where << ": the i32 proof substituted is " << render(subst)
            << ", the prover's own " << toString(p);
      }

      NewGuard g = materializeAt(p, use->getParentOfType<scf::ForOp>());
      const bool narrow = w < 32;
      for (int64_t k : narrow ? allValues(w) : boundaryValues(w, end)) {
        CanonicalEval sampled = canonicalMask(k, w, end, false);
        CanonicalEval exact = narrow ? canonicalMask(k, w, end, true) : sampled;
        if (narrow) {
          // The model against the interpreter, the sampler against both.
          auto run = oracle.run(am->module.get(), {k}, "mask");
          if (!run || exact.allTrue != every(*run, true) ||
              exact.anyTrue != !every(*run, false) ||
              exact.trips != static_cast<I128>(run->size()))
            log.mismatch(w, end, std::nullopt, {k}, "mask model");
          if (sampled.allTrue != exact.allTrue)
            log.mismatch(w, end, std::nullopt, {k}, "iteration sampler");
        }
        bool legacy = canonicalLegacy(k, w, end);
        bool newOk = holds(oracle, g, {k});
        bool unsound = (newOk && !exact.allTrue) ||
                       (g.verdict == BoundProof::Refuted && exact.anyTrue);
        log.record(w, end, std::nullopt, {k},
                   classify(legacy, exact.allTrue, newOk,
                            {{newOk, false, exact.exceedsWidth}}),
                   newOk, unsound, g.text);
      }

      if (w == 8 && end == 4) {
        // The largest i8 multiple of 4 is 124 = 127 - 3, so the wrap
        // guard loses nothing the legacy K % 4 == 0 && K > 4 admits.
        EXPECT_EQ(toString(p),
                  "Conditional{arg0 divisible by 4; arg0 >= 0; arg0 <= 124}");
        EXPECT_TRUE(canonicalLegacy(124, 8, 4) && holds(oracle, g, {124}));
      }
    }
  }
}

void LegacyImplicationTest::checkInvariant(const ImplicationCase &c,
                                           ImplicationLog &log) {
  using tt::intel::BoundProof;
  const LegacyFamily f = c.family;
  const SmallVector<Conjunct, 2> cjs = conjunctsOf(f, c.pred);
  const unsigned nargs = numArgs(f);
  const bool isBoundary = f == LegacyFamily::Boundary;
  const bool isAnd = f == LegacyFamily::AndRanges;

  if (f == LegacyFamily::Scalar) {
    // The legacy scalar form's only width, as real IR, every value.
    std::unique_ptr<AnalyzedModule> am =
        analyzeModule(ctx, invariantIR(f, c.pred, 1, 0, {}));
    ASSERT_TRUE(am);
    std::vector<MaskInstance> mi = proveAll(*am, 1, false);
    forEachTuple(allValues(1), nargs, [&](ArrayRef<int64_t> args) {
      checkInvariantPoint(log, cjs, 1, 0, std::nullopt, args, mi[0].guard, {});
    });
  }

  for (int64_t end : endsFor(f, 8)) {
    // The boundary check's bounds: every i8 value for the narrow model, and
    // the boundary values at i32 and i64. The other families have no bound.
    auto boundsFor = [&](unsigned w) {
      if (!isBoundary)
        return std::vector<int64_t>{0};
      std::vector<int64_t> out = allValues(8);
      llvm::append_range(out, boundaryValues(w, end));
      llvm::sort(out);
      out.erase(std::unique(out.begin(), out.end()), out.end());
      return out;
    };
    const std::vector<int64_t> bounds32 = boundsFor(32),
                               bounds64 = boundsFor(64);
    auto indexOf = [](const std::vector<int64_t> &v, int64_t x) {
      return static_cast<unsigned>(llvm::find(v, x) - v.begin());
    };
    std::unique_ptr<AnalyzedModule> am32 =
        analyzeModule(ctx, invariantIR(f, c.pred, 32, end, bounds32));
    std::unique_ptr<AnalyzedModule> am64 =
        analyzeModule(ctx, invariantIR(f, c.pred, 64, end, bounds64));
    ASSERT_TRUE(am32 && am64);
    std::vector<MaskInstance> m32 = proveAll(*am32, bounds32.size(), isAnd);
    std::vector<MaskInstance> m64 = proveAll(*am64, bounds64.size(), isAnd);

    // i32 and i64: real IR, the boundary values of every argument and bound.
    for (unsigned w : {32u, 64u}) {
      const std::vector<int64_t> &bounds = w == 32 ? bounds32 : bounds64;
      std::vector<MaskInstance> &masks = w == 32 ? m32 : m64;
      for (int64_t b :
           isBoundary ? boundaryValues(w, end) : std::vector<int64_t>{0}) {
        const MaskInstance &mi = masks[indexOf(bounds, b)];
        std::optional<int64_t> bound =
            isBoundary ? std::optional<int64_t>(b) : std::nullopt;
        forEachTuple(boundaryValues(w, end), nargs,
                     [&](ArrayRef<int64_t> args) {
                       checkInvariantPoint(log, cjs, w, end, bound, args,
                                           mi.guard, mi.sideGuards);
                     });
      }
    }

    // i4 and i8: the model. The substitution must reproduce the prover's own
    // i32 guard, and its own i64 guard where that is decided too.
    struct Narrow {
      std::optional<SmallVector<CondTemplate, 4>> t;
      SmallVector<SmallVector<CondTemplate, 4>, 2> sides;
      BoundProof::Verdict verdict;
      SmallVector<BoundProof::Verdict, 2> sideVerdicts;
      std::string why;
    };
    std::map<int64_t, Narrow> narrow; // by bound
    for (int64_t b : isBoundary ? allValues(8) : std::vector<int64_t>{0}) {
      const MaskInstance &i32 = m32[indexOf(bounds32, b)];
      const MaskInstance &i64 = m64[indexOf(bounds64, b)];
      std::string where = std::string(c.name) + " END=" + std::to_string(end) +
                          (isBoundary ? " c=" + std::to_string(b) : "");
      Narrow &n = narrow[b];
      auto mapOne = [&](const BoundProof &p32, const BoundProof &p64,
                        std::string &why) {
        auto t = templateOf(p32.conditions, end, why);
        if (!t)
          return t;
        EXPECT_TRUE(
            sameConditions(instantiate(*t, 32, end, *am32), p32.conditions))
            << where << ": the substitution does not reproduce "
            << toString(p32);
        EXPECT_EQ(p32.verdict, p64.verdict)
            << where << ": i32 " << toString(p32) << ", i64 " << toString(p64);
        if (p32.verdict == p64.verdict &&
            p64.verdict == BoundProof::ConditionallySatisfied) {
          auto subst = instantiate(*t, 64, end, *am64);
          EXPECT_TRUE(sameConditions(subst, p64.conditions))
              << where << ": the i32 proof substituted to i64 is "
              << render(subst) << ", the prover's own " << toString(p64);
        }
        return t;
      };
      n.verdict = i32.proof.verdict;
      n.t = mapOne(i32.proof, i64.proof, n.why);
      for (auto [s32, s64] : llvm::zip(i32.sides, i64.sides)) {
        std::string why;
        auto st = mapOne(s32, s64, why);
        if (!st) {
          n.t.reset();
          n.why = why;
          break;
        }
        n.sides.push_back(*st);
        n.sideVerdicts.push_back(s32.verdict);
      }
    }
    for (unsigned w : {4u, 8u}) {
      if (!llvm::is_contained(endsFor(f, w), end))
        continue;
      std::unique_ptr<AnalyzedModule> shell =
          analyzeModule(ctx, shellIR(w, nargs));
      ASSERT_TRUE(shell);
      Operation *ret = shell->func().getBody().front().getTerminator();
      auto narrowGuard = [&](const SmallVector<CondTemplate, 4> &t,
                             BoundProof::Verdict v) {
        BoundProof np;
        np.verdict = v;
        if (v == BoundProof::ConditionallySatisfied)
          np.conditions = instantiate(t, w, end, *shell);
        return materializeAt(np, ret);
      };
      for (int64_t b : isBoundary ? allValues(w) : std::vector<int64_t>{0}) {
        const Narrow &n = narrow[b];
        if (!n.t) {
          log.exclude(w, end, n.why);
          continue;
        }
        NewGuard g = narrowGuard(*n.t, n.verdict);
        SmallVector<NewGuard, 2> sideGuards;
        for (auto [st, sv] : llvm::zip(n.sides, n.sideVerdicts))
          sideGuards.push_back(narrowGuard(st, sv));
        std::optional<int64_t> bound =
            isBoundary ? std::optional<int64_t>(b) : std::nullopt;
        forEachTuple(allValues(w), nargs, [&](ArrayRef<int64_t> args) {
          checkInvariantPoint(log, cjs, w, end, bound, args, g, sideGuards);
        });
      }
    }
  }
}

TEST_P(LegacyImplicationTest, LegacyGuardImpliesNewGuard) {
  const ImplicationCase &c = GetParam();
  ImplicationLog log(c.name);
  if (c.family == LegacyFamily::Canonical)
    checkCanonical(log);
  else
    checkInvariant(c, log);
  log.finish(c.legacyVacuous, c.knownCoverageGap);
}

static std::string
implicationName(const ::testing::TestParamInfo<ImplicationCase> &info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Legacy, LegacyImplicationTest,
                         ::testing::ValuesIn(kImplicationCases),
                         implicationName);

//===----------------------------------------------------------------------===//
// Legacy counterexamples and the checker's own parts
//===----------------------------------------------------------------------===//

class LegacyImplicationParts : public OracleFixture {
protected:
  /// The boundary check's real proof at i32 or i64 for one bound, evaluated
  /// at one offset.
  bool boundaryNewGuard(Pred pred, unsigned w, int64_t end, int64_t c,
                        int64_t off) {
    std::unique_ptr<AnalyzedModule> am = analyzeModule(
        ctx, invariantIR(LegacyFamily::Boundary, pred, w, end, {c}));
    if (!am)
      return false;
    Operation *use = am->op("u0");
    NewGuard g = materializeAt(driverProof(*am, use, use->getOperand(0)),
                               use->getParentOfType<scf::ForOp>());
    return holds(oracle, g, {off});
  }
};

// Each legacy counterexample, at i32 and i64, through the code the
// parametrized test classifies with: legacy-unsound, never a conservative
// loss, and caught only because the sampler looks between the endpoints.
TEST_F(LegacyImplicationParts, LegacyCounterexamplesAreUnsound) {
  struct Example {
    const char *what;
    Pred pred;
    int64_t (*off)(unsigned w);
    int64_t (*c)(unsigned w);
  };
  const Example examples[] = {
      {"INT_MAX - 1 + [0..3] slt INT_MAX", Pred::slt,
       [](unsigned w) { return static_cast<int64_t>(smaxOf(w) - 1); },
       [](unsigned w) { return static_cast<int64_t>(smaxOf(w)); }},
      {"-1 + [0..3] uge 1", Pred::uge, [](unsigned) { return int64_t(-1); },
       [](unsigned) { return int64_t(1); }},
      {"INT_MAX + [0..3] sge INT_MIN + 1", Pred::sge,
       [](unsigned w) { return static_cast<int64_t>(smaxOf(w)); },
       [](unsigned w) { return static_cast<int64_t>(sminOf(w) + 1); }},
  };
  for (const Example &ex : examples)
    for (unsigned w : {32u, 64u}) {
      int64_t off = ex.off(w), c = ex.c(w);
      Conjunct cj{Form::OffsetLaneVsConst, ex.pred, 0};
      ConjunctEval e = evalConjunct(cj, {wrapToWidth(off, w)}, w, 4, c);
      std::string where = std::string(ex.what) + " at i" + std::to_string(w);
      EXPECT_EQ(e.lanes, (SmallVector<bool, 8>{true, false, true, true}))
          << where;
      EXPECT_FALSE(e.sampledAllTrue) << where;
      EXPECT_TRUE(e.lanes.front() && e.lanes.back())
          << where << ": the endpoints alone would pass";
      EXPECT_TRUE(e.legacy) << where;
      EXPECT_FALSE(boundaryNewGuard(ex.pred, w, 4, c, off)) << where;
      EXPECT_EQ(classify(e.legacy, false, false, {{false, e.kindI, e.kindII}}),
                PointClass::LegacyUnsound)
          << where;
    }

  // i8, offset 125, make_range(0, 4), bound 4: the legacy guard computes
  // -128 < 4 and passes while lanes 0-2 are false.
  ConjunctEval e = evalConjunct({Form::OffsetLaneVsConst, Pred::slt, 0},
                                {wrapToWidth(125, 8)}, 8, 4, 4);
  EXPECT_EQ(e.lanes, (SmallVector<bool, 8>{false, false, false, true}));
  EXPECT_TRUE(e.legacy);
}

// The paragraph's two conservative losses: sound legacy points the new guard
// rejects, each carrying its kind.
TEST_F(LegacyImplicationParts, ConservativeLossesCarryTheirKind) {
  // (ii) (off + [0..3]) sle INT32_MAX at off = INT32_MAX: every wrapped lane
  // passes, the new proof needs off <= INT32_MAX - 3.
  int64_t off = INT32_MAX;
  ConjunctEval wrap = evalConjunct({Form::OffsetLaneVsConst, Pred::sle, 0},
                                   {wrapToWidth(off, 32)}, 32, 4, INT32_MAX);
  EXPECT_TRUE(wrap.legacy &&
              llvm::all_of(wrap.lanes, [](bool b) { return b; }));
  EXPECT_FALSE(boundaryNewGuard(Pred::sle, 32, 4, INT32_MAX, off));
  EXPECT_EQ(classify(true, true, false, {{false, wrap.kindI, wrap.kindII}}),
            PointClass::LossWrap);

  // (i) [0..3] ult N with N's sign bit set.
  ConjunctEval uns = evalConjunct({Form::LaneVsArg, Pred::ult, 0},
                                  {wrapToWidth(-1, 32)}, 32, 4, 0);
  EXPECT_TRUE(uns.legacy && llvm::all_of(uns.lanes, [](bool b) { return b; }));
  EXPECT_EQ(classify(true, true, false, {{false, uns.kindI, uns.kindII}}),
            PointClass::LossUnsigned);
  // Neither kind: the same loss on a signed predicate with no wrap fails.
  EXPECT_EQ(classify(true, true, false, {{false, false, false}}),
            PointClass::LossUnclassified);
}

// Both sides of a falling discontinuity, signed at INT_MIN and unsigned at 0.
TEST_F(LegacyImplicationParts, SamplerSeesBothSidesOfEveryWrap) {
  // i8: -126, -127, -128, 127, 126 wraps between j = 2 and j = 3.
  auto s = discontinuitySamples(-126, -1, 5, 8, false);
  EXPECT_TRUE(llvm::is_contained(s, I128(2)) && llvm::is_contained(s, I128(3)));
  // 1, 0, 255, 254 unsigned wraps between j = 1 and j = 2.
  s = discontinuitySamples(1, -1, 4, 8, true);
  EXPECT_TRUE(llvm::is_contained(s, I128(1)) && llvm::is_contained(s, I128(2)));
  // No wrap: the endpoints only.
  s = discontinuitySamples(0, 1, 8, 8, false);
  EXPECT_EQ(s.size(), 2u);
}

// Only the two named wrap constants take the width; a width-dependent constant
// of any other shape keeps the family out of the narrow model.
TEST_F(LegacyImplicationParts, SubstitutionCarriesOnlyTheNamedConstants) {
  using namespace tt::intel;
  std::unique_ptr<AnalyzedModule> am = analyzeModule(ctx, shellIR(32, 1)),
                                  shell = analyzeModule(ctx, shellIR(8, 1));
  ASSERT_TRUE(am && shell);
  AffineForm x = AffineForm::symbol(
      am->prover->symbolFor(SymbolKind::KernelArg, am->arg(0)));
  SmallVector<BoundCondition, 4> conds = {
      {x, BoundGoal::DivisibleBy, 4, ConditionKind::Fact},
      {x, BoundGoal::NonNegative, 0, ConditionKind::Precondition},
      {x, BoundGoal::AtLeast, 5, ConditionKind::Fact},
      {x, BoundGoal::AtMost, INT32_MAX - 3, ConditionKind::Guard},
      {x, BoundGoal::AtLeast, INT32_MIN, ConditionKind::Guard}};
  std::string why;
  auto t = templateOf(conds, /*end=*/4, why);
  ASSERT_TRUE(t) << why;
  EXPECT_TRUE(sameConditions(instantiate(*t, 32, 4, *am), conds));
  EXPECT_EQ(render(instantiate(*t, 8, 4, *shell)),
            "{arg0 divisible by 4; arg0 >= 0; arg0 >= 5; arg0 <= 124; "
            "arg0 >= -128}");

  SmallVector<BoundCondition, 1> other = {
      {x, BoundGoal::AtMost, INT32_MAX - 7, ConditionKind::Guard}};
  EXPECT_FALSE(templateOf(other, 4, why));
}

} // namespace
