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
// no-overflow condition of design 4.1 and a false `llvm.intr.assume`. A launch
// that is skipped proves nothing, so the accounting below counts how many
// launches were valid, how many had a true guard, and how many mask elements
// were actually evaluated under a true guard; a case whose safety check never
// runs fails.
//
// Each case names exactly one verdict, so a prover that regresses to a vacuous
// Unknown fails here rather than silently stopping to prove anything (design
// 6). The oracle is deliberately independent of the prover: it knows the IR's
// wrapping semantics and nothing about affine forms.
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
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
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
/// not itself subject to the wrapping it is testing.
using I128 = __int128;

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
  return APInt(w, static_cast<uint64_t>(static_cast<unsigned __int128>(v)),
               /*isSigned=*/false, /*implicitTrunc=*/true);
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
  /// no-overflow condition of design 4.1, a false `llvm.intr.assume`, or
  /// undefined arithmetic (division by zero, an out-of-range shift) - which
  /// the caller skips instead of checking. An operation outside the modelled
  /// subset fails the test rather than skipping silently.
  std::optional<std::vector<std::vector<bool>>>
  run(ModuleOp m, ArrayRef<int64_t> argValues, StringRef maskName) {
    env.clear();
    masks.clear();
    mask = maskName;

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

private:
  /// `Skip` ends the launch without a verdict (undefined or assume-violating);
  /// `Fail` means the oracle does not model something it met, and has already
  /// failed the test.
  enum class Status { Ok, Skip, Fail };

  /// A malformed fixture must fail rather than hang: no modelled loop over an
  /// i8 iteration space can run this many times.
  static constexpr I128 kMaxIterations = 4096;

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
      return uns ? static_cast<I128>(a.getZExtValue())
                 : static_cast<I128>(a.getSExtValue());
    };
    I128 lb = read(forOp.getLowerBound());
    I128 ub = read(forOp.getUpperBound());
    I128 step = read(forOp.getStep());
    if (step <= 0) {
      ADD_FAILURE() << "oracle: scf.for step is not positive";
      return Status::Fail;
    }

    I128 n = ub > lb ? (ub - lb + step - 1) / step : 0;
    // Loop contract of design 4.1: the value one step past the last iteration
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
  /// symbol a condition can name (design 4.4).
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
    R"(   // Conditional{arg0 >= -127}: Task 5 LoopBoundWrapObligation
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
static const char *kE1_I8Scalars =
    R"(      // Conditional{arg0 divisible by 4}; 256 launches
  tt.func @e1(%n8: i8) {
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
static const char *kE2_I8Scalars =
    R"(      // Conditional{arg0 divisible by 4; arg0 >= 0; arg0 <= 124}
  tt.func @e2(%K: i8) {
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
    R"(    // Conditional{arg0 <= 99; arg0 >= 0}: 4e, normalized, then the ult obligation
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
// Each case names exactly one verdict, so a vacuous Unknown fails (spec 6).
// Safety is then checked by execution for every verdict except Unknown, which
// makes no claim; Unknown cases are verdict-only, so the oracle needs no
// tt.load support for two_axes_load.
// `incr` is the increment whose candidates the case needs: "1b" cases use only
// 4a, 4b and obligation guards; "1c" cases need 4c, 4d or 4e.
struct Case {
  const char *name;
  std::string ir;
  const char *mask;
  std::vector<V> allowed;
  const char *incr = "1b";
  bool vacuous = false;
};
static const Case kCases[] = {
    {"e1_i8", kE1_I8Scalars, "mask", {V::ConditionallySatisfied}},
    {"e2_i8", kE2_I8Scalars, "mask", {V::ConditionallySatisfied}},
    {"two_axes", kSameLaneTwoAxes, "cmp", {V::Unknown}},
    {"two_axes_load",
     kSameLoadTwoAxes,
     "cmp",
     {V::Unknown}}, // loaded tensor on two axes
    {"wrap_i8",
     kNarrowRangeWrapI8,
     "cmp",
     {V::ConditionallySatisfied},
     "1b",
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
     "1c"}, // 4e bounds the residual, then the ult obligations
    {"extui",
     kExtUII8,
     "cmp",
     {V::ConditionallySatisfied},
     "1c"}, // 4c proves the operand sign; the obligation alone is not search
            // evidence
    {"zero_trip", kEmptyLoopI8, "mask", {V::ConditionallySatisfied}},
    {"product_wrap",
     kProductSignI8,
     "cmp",
     {V::ConditionallySatisfied},
     "1c"}, // 4c guards p itself
    {"product_wrap_i4",
     kProductSignI4,
     "cmp",
     {V::ConditionallySatisfied},
     "1c"},
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

// Two suites, so --gtest_filter selects an increment and gtest reports the
// names: the 1b gate runs `--gtest_filter='Incr1b/*'`, which registers only
// the cases its candidates can prove. `incr` is the single source of both
// lists, so a case cannot be in neither or both.
static std::vector<Case> casesFor(StringRef incr) {
  std::vector<Case> out;
  for (const Case &c : kCases)
    if (c.incr == incr)
      out.push_back(c);
  return out;
}

/// Names each instantiation after its case, so a failure reads as
/// `Incr1b/SymbolicBoundsOracleTest.VerdictMatchesExhaustiveExecution/e1_i8`.
static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Incr1b, SymbolicBoundsOracleTest,
                         ::testing::ValuesIn(casesFor("1b")), caseName);
INSTANTIATE_TEST_SUITE_P(Incr1c, SymbolicBoundsOracleTest,
                         ::testing::ValuesIn(casesFor("1c")), caseName);

//===----------------------------------------------------------------------===//
// Guard arithmetic
//===----------------------------------------------------------------------===//

class SymbolicBoundsGuardTest : public OracleFixture {};

// Guard arithmetic near the i64 fit limit (spec 6), independent of the prover.
// k = 2^55 takes the fast path (2^55*2^7 < 2^63); 2^56 is the first checked
// one. The guard must equal k*a >= t in exact arithmetic, and be false wherever
// k*a overflows i64 (spec 4.4). SymbolicBoundsGuardTest is the oracle fixture
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
        __int128 exact = static_cast<__int128>(k) * a;
        bool fits = exact >= INT64_MIN && exact <= INT64_MAX;
        EXPECT_EQ(oracle.evalGuard(guard, {a}), fits && exact >= t)
            << k << " " << t << " " << a;
      }
    }
}

} // namespace
