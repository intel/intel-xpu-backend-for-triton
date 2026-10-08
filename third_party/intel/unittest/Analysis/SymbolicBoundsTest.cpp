//===- SymbolicBoundsTest.cpp ---------------------------------------------===//
//
// Unit tests for triton::intel::SymbolicBoundsProver, the goal-directed
// symbolic bounds prover behind the symbolic mask removal in RemoveMasks.
//
// Every normalization rule and candidate condition is a soundness claim, and a
// wrong one lets a consumer drop a mask whose lanes are not all true. A lit
// test can only observe the enclosing pass's aggregate decision, so each rule
// is exercised here directly, in both directions: what it proves, and what it
// declines to prove.
//
// Cases are rooted at function arguments wherever the point is that the range
// analysis cannot answer by itself; otherwise the prover would answer from the
// range analysis alone and the rule under test would never run.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/SymbolicBounds.h"
#include "intel/include/Analysis/Range.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>
#include <string>
#include <tuple>

using namespace mlir;
namespace tt = mlir::triton;

namespace {

class SymbolicBoundsTest : public ::testing::Test {
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

  /// Renders the normalized affine form of `v`.
  std::string norm(Value v) {
    SmallVector<tt::intel::Obligation, 4> obls;
    return tt::intel::toString(prover->normalize(v, at(v), obls));
  }

  /// The proof of the comparison named `loc("<name>")`, for assertions about
  /// the conditions or the facts it used.
  tt::intel::BoundProof proof(Value cmp) {
    auto op = cast<arith::CmpIOp>(cmp.getDefiningOp());
    return prover->prove(op.getPredicate(), op.getLhs(), op.getRhs(), at(cmp));
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

  /// Renders the verdict of the comparison named `loc("<name>")`.
  std::string verdict(Value cmp) {
    auto op = cast<arith::CmpIOp>(cmp.getDefiningOp());
    return tt::intel::toString(
        prover->prove(op.getPredicate(), op.getLhs(), op.getRhs(), at(cmp)));
  }

  /// Returns argument `idx` of the (single) function in the parsed IR.
  Value arg(unsigned idx) {
    Value found;
    module->walk([&](tt::FuncOp funcOp) { found = funcOp.getArgument(idx); });
    EXPECT_TRUE(found) << "no function argument " << idx;
    return found;
  }

protected:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<DataFlowSolver> solver;
  std::unique_ptr<tt::intel::SymbolicBoundsProver> prover;
};

TEST_F(SymbolicBoundsTest, ScalarAffine) {
  parse(R"(
    tt.func @f(%a: i32, %b: i32) {
      %c3 = arith.constant 3 : i32
      %c5 = arith.constant 5 : i32
      %m = arith.muli %a, %c3 : i32 loc("m")
      %s = arith.addi %m, %b : i32 loc("s")
      %d = arith.subi %s, %c5 : i32 loc("d")
      %p = arith.muli %a, %b : i32 loc("p")
      tt.return
    })");
  EXPECT_EQ(norm(get("d")), "3*arg0 + arg1 - 5");
  EXPECT_EQ(norm(get("p")), "opaque(p)"); // product of two symbols
}

TEST_F(SymbolicBoundsTest, SameLaneSameAxisCancels) {
  parse(R"(
    tt.func @f(%n: i32) {
      %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %c1 = arith.constant dense<1> : tensor<64xi32>
      %r1 = arith.addi %r, %c1 : tensor<64xi32> loc("r1")
      %cmp = arith.cmpi slt, %r, %r1 : tensor<64xi32> loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Satisfied"); // r < r + 1, element-wise
}

TEST_F(SymbolicBoundsTest, SameLaneDifferentAxesDoesNotCancel) {
  parse(R"(
    tt.func @f() {
      %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %col = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %row = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %cb = tt.broadcast %col : tensor<64x1xi32> -> tensor<64x64xi32>
      %rb = tt.broadcast %row : tensor<1x64xi32> -> tensor<64x64xi32>
      %cmp = arith.cmpi slt, %cb, %rb : tensor<64x64xi32> loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Unknown"); // never Refuted
}

// The inductor reduction shape: `r + lane < rnumel` over a loop
// `0 to rnumel step 64`.
static const char *kReductionLoop = R"(
  tt.func @reduction_loop(%ptr: !tt.ptr<f32>, %rnumel: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
    scf.for %r = %c0 to %rnumel step %c64 : i32 {
      %rs = tt.splat %r : i32 -> tensor<64xi32>
      %idx = arith.addi %rs, %lane : tensor<64xi32> loc("idx")
      %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";

TEST_F(SymbolicBoundsTest, ReductionLoopConditionalOnExactLoopEnd) {
  parse(kReductionLoop);
  EXPECT_EQ(verdict(get("mask")), "Conditional{arg1 divisible by 64}");
}

TEST_F(SymbolicBoundsTest, ConstantBoundsSatisfied) {
  parse(R"(
    tt.func @f(%ptr: !tt.ptr<f32>) {
      %c0 = arith.constant 0 : i32
      %c32 = arith.constant 32 : i32
      %c128 = arith.constant 128 : i32
      %c4096 = arith.constant dense<4096> : tensor<32xi32>
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
      scf.for %i = %c0 to %c128 step %c32 : i32 {
        %is = tt.splat %i : i32 -> tensor<32xi32>
        %idx = arith.addi %is, %lane : tensor<32xi32>
        %m1 = arith.cmpi slt, %idx, %c4096 : tensor<32xi32> loc("m1")
        %m2 = arith.cmpi sge, %idx, %c4096 : tensor<32xi32> loc("m2")
        scf.yield
      }
      tt.return
    })");
  EXPECT_EQ(verdict(get("m1")), "Satisfied"); // max 96 + 31 = 127 < 4096
  EXPECT_EQ(verdict(get("m2")), "Refuted");   // never >= 4096
}

TEST_F(SymbolicBoundsTest, UnsignedLoopNeedsSignedRepresentability) {
  std::string ir = kReductionLoop;
  ir.replace(ir.find("scf.for %r"), 10, "scf.for unsigned %r");
  parse(ir);
  // Reading unsigned bounds as signed needs three preconditions:
  // NonNegative(lb) discharges because lb = 0 is a constant; NonNegative(ub)
  // and AtMost(ub, INT_MAX - step + 1) remain.
  // Order: fact, then preconditions.
  EXPECT_EQ(verdict(get("mask")),
            "Conditional{arg1 divisible by 64; arg1 >= 0; arg1 <= 2147483584}");
}

TEST_F(SymbolicBoundsTest, OffsetIterArgNeedsRepresentability) {
  // An i8 loop lb=120..ub=124 step 1 is well-defined - the exit value 124 fits
  // - but an iter_arg started at lb + 5 takes 125, 126, 127, -128. Treated as
  // IV + 5 with no obligation it would prove `o >= 0`; the obligation's
  // hi = 123 + 5 = 128 is a constant out of range, so no guard can help.
  parse(R"(
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
    })");
  EXPECT_EQ(verdict(get("cmp")), "Unknown");
}

TEST_F(SymbolicBoundsTest, DepthCapIsUnknown) {
  // Fully normalized, v20 = x + 20 and `v20 < x` is Refuted (hi(d) = -20). The
  // chain exceeds the depth budget, which degrades the query to Unknown, never
  // to the Refuted a truncated traversal would also reach.
  std::string ir = "tt.func @f(%a: i8) {\n  %c1 = arith.constant 1 : i32\n"
                   "  %x = arith.extsi %a : i8 to i32\n";
  for (int i = 1; i <= 20; ++i)
    ir += "  %v" + std::to_string(i) + " = arith.addi %" +
          (i == 1 ? std::string("x") : "v" + std::to_string(i - 1)) +
          ", %c1 : i32\n";
  ir +=
      "  %cmp = arith.cmpi slt, %v20, %x : i32 loc(\"cmp\")\n  tt.return\n}\n";
  parse(ir);
  EXPECT_EQ(verdict(get("cmp")), "Unknown");
}

TEST_F(SymbolicBoundsTest, MemoHitRespectsDepthBudget) {
  // v10 is normalized first at depth 0 (height 10, within the cap), then
  // reached again below a 10-deep chain: 10 + 10 > 16 must exhaust, so a
  // subtree memoized near the root cannot bypass the cap below a deep one.
  std::string ir = "tt.func @f(%a: i8) {\n  %c1 = arith.constant 1 : i32\n"
                   "  %v0 = arith.extsi %a : i8 to i32\n";
  for (int i = 1; i <= 20; ++i)
    ir += "  %v" + std::to_string(i) + " = arith.addi %v" +
          std::to_string(i - 1) + ", %c1 : i32\n";
  ir += "  %near = arith.cmpi slt, %v10, %v0 : i32 loc(\"near\")\n"
        "  %deep = arith.cmpi slt, %v20, %v0 : i32 loc(\"deep\")\n  "
        "tt.return\n}\n";
  parse(ir);
  EXPECT_EQ(verdict(get("near")), "Refuted"); // v10 = a + 10 < a never holds
  EXPECT_EQ(verdict(get("deep")), "Unknown"); // memo hit on v10 at depth 10
}

TEST_F(SymbolicBoundsTest, NormalizationRules) {
  // One case per remaining normalization rule.
  parse(R"(
    tt.func @f(%a: i32, %b: index) {
      %np = tt.get_num_programs x : i32 loc("np")
      %w = arith.index_cast %a : i32 to index loc("w")
      %n = arith.index_cast %b : index to i32 loc("n")
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %c2 = arith.constant dense<2> : tensor<4xi32>
      %q = arith.divsi %r, %c2 : tensor<4xi32>
      %qc = tt.expand_dims %q {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %qr = tt.expand_dims %q {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %qcb = tt.broadcast %qc : tensor<4x1xi32> -> tensor<4x4xi32>
      %qrb = tt.broadcast %qr : tensor<1x4xi32> -> tensor<4x4xi32>
      %qcmp = arith.cmpi slt, %qcb, %qrb : tensor<4x4xi32> loc("qcmp")
      %t = tt.trans %qcb {order = array<i32: 1, 0>} : tensor<4x4xi32> -> tensor<4x4xi32> loc("t")
      %rs = tt.reshape %r : tensor<4xi32> -> tensor<2x2xi32> loc("rs")
      %j = tt.join %r, %r : tensor<4xi32> -> tensor<4x2xi32> loc("j")
      %lo, %hi = tt.split %j : tensor<4x2xi32> -> tensor<4xi32> loc("s")
      %lh = arith.addi %lo, %hi : tensor<4xi32> loc("lh")
      %hl = arith.addi %hi, %lo : tensor<4xi32> loc("hl")
      tt.return
    })");
  EXPECT_EQ(norm(get("np")), "np");       // NumPrograms, rendered by loc name
  EXPECT_EQ(norm(get("w")), "arg0");      // widening index_cast passes through
  EXPECT_EQ(norm(get("n")), "opaque(n)"); // narrowing index_cast truncates
  EXPECT_EQ(norm(get("t")), "opaque(t)"); // permutations are Opaque
  EXPECT_EQ(norm(get("rs")), "opaque(rs)");
  EXPECT_EQ(norm(get("j")), "opaque(j)");
  // Both split results, distinct symbols; the result number breaks the tie in
  // the total symbol order, so the two sums render identically.
  EXPECT_EQ(norm(get("lh")), "opaque(s#0) + opaque(s#1)");
  EXPECT_EQ(norm(get("hl")), norm(get("lh")));
  EXPECT_EQ(verdict(get("qcmp")), "Unknown"); // quotient on two axes: no cancel
}

TEST_F(SymbolicBoundsTest, TermBudgetAndDynamicStepAreUnknown) {
  // A chain of `n` distinct symbols summed into `s`, then `s >= s`, which is
  // true for any n. The operands are i8 widened to i32 so every addi's wrap
  // obligation discharges statically from the operand ranges: on i32 operands
  // each addi contributes two wrap guards instead and kMaxGuards is exhausted
  // at about six symbols, which would make this Unknown well inside kMaxTerms
  // and leave the term budget untested.
  auto chainOf = [](int n) {
    std::string ir = "tt.func @f(";
    for (int i = 0; i < n; ++i)
      ir += (i ? ", %b" : "%b") + std::to_string(i) + ": i8";
    ir += ") {\n";
    for (int i = 0; i < n; ++i)
      ir += "  %a" + std::to_string(i) + " = arith.extsi %b" +
            std::to_string(i) + " : i8 to i32\n";
    ir += "  %s0 = arith.addi %a0, %a1 : i32\n";
    for (int i = 1; i < n - 1; ++i)
      ir += "  %s" + std::to_string(i) + " = arith.addi %s" +
            std::to_string(i - 1) + ", %a" + std::to_string(i + 1) + " : i32\n";
    std::string last = "%s" + std::to_string(n - 2);
    ir += "  %cmp = arith.cmpi sge, " + last + ", " + last +
          " : i32 loc(\"cmp\")\n  tt.return\n}\n";
    return ir;
  };
  // The budget is pinned from both sides, so raising kMaxTerms fails the test.
  parse(chainOf(tt::intel::SymbolicBoundsProver::kMaxTerms));
  EXPECT_EQ(verdict(get("cmp")), "Satisfied");
  parse(chainOf(tt::intel::SymbolicBoundsProver::kMaxTerms + 1));
  EXPECT_EQ(verdict(get("cmp")), "Unknown");

  // A non-constant step makes the IV Opaque.
  std::string dyn = kReductionLoop;
  dyn.replace(dyn.find("%rnumel: i32)"), 13, "%rnumel: i32, %st: i32)");
  dyn.replace(dyn.find("step %c64"), 9, "step %st");
  parse(dyn);
  EXPECT_EQ(verdict(get("mask")), "Unknown");
}

TEST_F(SymbolicBoundsTest, RefutedOnlyOnConstantHi) {
  // pid >= 0 from the range analysis, so hi(d) = -pid <= 0 < 1 by sign
  // reasoning; refutation needs a constant hi(d).
  parse(R"(
    tt.func @f() {
      %c0 = arith.constant 0 : i32
      %pid = tt.get_program_id x : i32
      %cmp = arith.cmpi slt, %pid, %c0 : i32 loc("cmp")
      tt.return
    })");
  EXPECT_NE(verdict(get("cmp")), "Refuted");
}

TEST_F(SymbolicBoundsTest, OverflowedDifferenceIsUnknown) {
  // d = rhs - lhs overflows int64 in both directions. Read through the
  // wrapped value, `a` would be a false Satisfied and `b` a false Refuted.
  parse(R"(
    tt.func @f() {
      %max = arith.constant 9223372036854775807 : i64
      %min = arith.constant -9223372036854775808 : i64
      %a = arith.cmpi slt, %max, %min : i64 loc("a")
      %b = arith.cmpi slt, %min, %max : i64 loc("b")
      tt.return
    })");
  EXPECT_EQ(verdict(get("a")), "Unknown");
  EXPECT_EQ(verdict(get("b")), "Unknown");
}

TEST_F(SymbolicBoundsTest, NegativeStartLane) {
  // start = -4: read through the unsigned getters the lane would bound as
  // [4294967292, 4294967295] and `r >= 0` would be Satisfied.
  parse(R"(
    tt.func @f() {
      %r = tt.make_range {start = -4 : i32, end = 0 : i32} : tensor<4xi32>
      %z = arith.constant dense<0> : tensor<4xi32>
      %cmp = arith.cmpi sge, %r, %z : tensor<4xi32> loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Refuted"); // every lane is negative
}

// The tutorial-03 K loop, with a cdiv upper bound.
static const char *kCdivKLoop = R"(
  tt.func @cdiv_k_loop(%K: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c63 = arith.constant 63 : i32
    %c64 = arith.constant 64 : i32
    %num = arith.addi %K, %c63 : i32 loc("num")
    %q = arith.divsi %num, %c64 : i32 loc("q")
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    scf.for %k = %c0 to %q step %c1 : i32 {
      %k64 = arith.muli %k, %c64 : i32 loc("k64")
      %rem = arith.subi %K, %k64 : i32 loc("rem")
      %rs = tt.splat %rem : i32 -> tensor<64xi32>
      %mask = arith.cmpi slt, %lane, %rs : tensor<64xi32> loc("mask")
      scf.yield
    }
    tt.return
  })";

TEST_F(SymbolicBoundsTest, CdivKLoopConditionalOnExactCdivWithPrecondition) {
  parse(kCdivKLoop);
  // A prefix check: CdivKLoopGainsCdivNumeratorGuard pins the full condition
  // list, including the K + 63 wrap guard.
  EXPECT_EQ(verdict(get("mask"))
                .rfind("Conditional{arg0 divisible by 64; arg0 >= 0", 0),
            0u);
}

TEST_F(SymbolicBoundsTest, CdivKLoopGainsCdivNumeratorGuard) {
  parse(kCdivKLoop);
  // The cdiv facts are used, so the numerator K + 63 must fit i32: the `X + c
  // - 1` addition is never traversed (the facts are stated about X), so its
  // wrap obligation is recorded explicitly or this guard would be lost.
  EXPECT_EQ(verdict(get("mask")),
            "Conditional{arg0 divisible by 64; arg0 >= 0; arg0 <= 2147483584}");
}

TEST_F(SymbolicBoundsTest, RemainderBounds) {
  parse(R"(
    tt.func @f(%x: i32) {
      %c64 = arith.constant 64 : i32
      %c0 = arith.constant 0 : i32
      %r = arith.remsi %x, %c64 : i32 loc("r")
      %lt = arith.cmpi slt, %r, %c64 : i32 loc("lt")
      %ge = arith.cmpi sge, %r, %c0 : i32 loc("ge")
      tt.return
    })");
  EXPECT_EQ(verdict(get("lt")), "Conditional{arg0 >= 0}");
  // Precondition retained even though it does not change lo(d).
  EXPECT_EQ(verdict(get("ge")), "Conditional{arg0 >= 0}");
}

TEST_F(SymbolicBoundsTest, VaryingQuotientBounds) {
  // X is loop-varying with range [100, 101]; q = X / 64 is 1, so q < 50 holds.
  // Bounding q by its stored dividend's range would refute it.
  parse(R"(
    tt.func @f(%p: !tt.ptr<i32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c50 = arith.constant 50 : i32
      %c64 = arith.constant 64 : i32
      %c100 = arith.constant 100 : i32
      %c101 = arith.constant 101 : i32
      scf.for %i = %c0 to %n step %c1 : i32 {
        %x = tt.load %p : !tt.ptr<i32>
        %ge = arith.cmpi sge, %x, %c100 : i32
        llvm.intr.assume %ge : i1
        %le = arith.cmpi sle, %x, %c101 : i32
        llvm.intr.assume %le : i1
        %q = arith.divsi %x, %c64 : i32
        %cmp = arith.cmpi slt, %q, %c50 : i32 loc("cmp")
        scf.yield
      }
      tt.return
    })");
  EXPECT_NE(verdict(get("cmp")), "Refuted");
}

TEST_F(SymbolicBoundsTest, XPlusOneGreaterThanXIsUnknown) {
  parse(R"(
    tt.func @f(%x: i32) {
      %c1 = arith.constant 1 : i32
      %y = arith.addi %x, %c1 : i32 loc("y")
      %cmp = arith.cmpi sgt, %y, %x : i32 loc("cmp")
      tt.return
    })");
  // Nothing bounds x, so the wrap obligation on y stays open; a Conditional
  // carrying x <= INT32_MAX - 1 is acceptable, Satisfied is not.
  EXPECT_NE(verdict(get("cmp")), "Satisfied");
  EXPECT_NE(verdict(get("cmp")), "Refuted");
}

TEST_F(SymbolicBoundsTest, NarrowResultRangeIsNotANoWrapProof) {
  parse(R"(
    tt.func @f(%a: i8) {
      %c100 = arith.constant 100 : i8
      %c110 = arith.constant 110 : i8
      %ge = arith.cmpi sge, %a, %c100 : i8
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %a, %c110 : i8
      llvm.intr.assume %le : i1
      %y = arith.addi %a, %c100 : i8 loc("y")
      %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
      tt.return
    })");
  // The range analysis reports y in [-56, -46], the correct range of the
  // WRAPPED result; d = 100 would say Satisfied without obligations. hi(y) is
  // 210 > 127 from the operands, so it must not.
  EXPECT_NE(verdict(get("cmp")), "Satisfied");
}

TEST_F(SymbolicBoundsTest, ExhaustedObligationBoundIsUnknown) {
  // y = 2*iv wraps in i64 for iv in [2^62, 2^62 + 2): both products are
  // negative and `y > iv` is false, although d = iv >= 2^62. Bounding the
  // muli obligation overflows, so the query must end Unknown.
  parse(R"(
    tt.func @f() {
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %lb = arith.constant 4611686018427387904 : i64
      %ub = arith.constant 4611686018427387906 : i64
      scf.for %iv = %lb to %ub step %c1 : i64 {
        %y = arith.muli %iv, %c2 : i64
        %cmp = arith.cmpi sgt, %y, %iv : i64 loc("cmp")
        scf.yield
      }
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Unknown");
}

TEST_F(SymbolicBoundsTest, ReductionLoopWrapDischargedByLoopContract) {
  parse(kReductionLoop);
  // r + lane discharges from the operand bounds under the final candidate
  // set: hi = (rnumel - 64) + 63 = rnumel - 1 <= INT32_MAX - 1. No guard.
  EXPECT_EQ(verdict(get("mask")), "Conditional{arg1 divisible by 64}");
}

TEST_F(SymbolicBoundsTest, LoopBoundWrapObligation) {
  // ub = n - 1 wraps for n = -128 (ub = 127). Read mathematically, hi(iv) is
  // n - 2 and `iv < n` is Satisfied, yet for n = -128 the loop runs 0..126
  // with the mask false. The subi's obligation from bounding the IV guards it.
  parse(R"(
    tt.func @f(%n: i8) {
      %c0 = arith.constant 0 : i8
      %c1 = arith.constant 1 : i8
      %ub = arith.subi %n, %c1 : i8
      scf.for %iv = %c0 to %ub step %c1 : i8 {
        %mask = arith.cmpi slt, %iv, %n : i8 loc("mask")
        scf.yield
      }
      tt.return
    })");
  EXPECT_NE(verdict(get("mask")), "Satisfied");
}

TEST_F(SymbolicBoundsTest, PreLoopTensorLoadCannotBeGuarded) {
  // A tensor loaded before the loop is loop-invariant but not a scalar, so it
  // may never become a condition subject: Unknown, not Conditional.
  parse(R"(
    tt.func @f(%p: !tt.ptr<i32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
      %pp = tt.addptr %ps, %lane : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
      %t = tt.load %pp : tensor<64x!tt.ptr<i32>>
      %ns = tt.splat %n : i32 -> tensor<64xi32>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %is = tt.splat %i : i32 -> tensor<64xi32>
        %idx = arith.addi %is, %t : tensor<64xi32>
        %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
        scf.yield
      }
      tt.return
    })");
  EXPECT_EQ(verdict(get("mask")), "Unknown");
}

TEST_F(SymbolicBoundsTest, TensorObligationGuardIsUnknown) {
  // extui and extsi of the same tensor cancel in d, so the direct path reaches
  // the verdict with no candidate; extui's NonNegative obligation is
  // tensor-valued and cannot be guarded, so Unknown, not Satisfied.
  parse(R"(
    tt.func @f(%p: !tt.ptr<i32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
      %pp = tt.addptr %ps, %lane : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
      %t = tt.load %pp : tensor<64x!tt.ptr<i32>>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %u = arith.extui %t : tensor<64xi32> to tensor<64xi64>
        %s = arith.extsi %t : tensor<64xi32> to tensor<64xi64>
        %mask = arith.cmpi sge, %u, %s : tensor<64xi64> loc("mask")
        scf.yield
      }
      tt.return
    })");
  EXPECT_EQ(verdict(get("mask")), "Unknown");
}

TEST_F(SymbolicBoundsTest, AssumeUnderIfDoesNotReachLoop) {
  parse(R"(
    tt.func @f(%ptr: !tt.ptr<f32>, %n: i32, %flag: i1) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %r = arith.remsi %n, %c64 : i32
      %cmp = arith.cmpi eq, %r, %c0 : i32
      scf.if %flag {
        llvm.intr.assume %cmp : i1
        scf.yield
      }
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ns = tt.splat %n : i32 -> tensor<64xi32>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %is = tt.splat %i : i32 -> tensor<64xi32>
        %idx = arith.addi %is, %lane : tensor<64xi32>
        %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
        scf.yield
      }
      tt.return
    })");
  // The divisibility fact is out of scope, so the condition stays runtime.
  EXPECT_EQ(verdict(get("mask")), "Conditional{arg1 divisible by 64}");
}

TEST_F(SymbolicBoundsTest, DominatingDivisibilityAssumeSatisfies) {
  // The assume dominates the loop, so the exact-loop-end candidate's
  // DivisibleBy(n, 64) is established by the fact and emitted as no runtime
  // condition.
  parse(R"(
    tt.func @f(%ptr: !tt.ptr<f32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %r = arith.remsi %n, %c64 : i32
      %cmp = arith.cmpi eq, %r, %c0 : i32
      llvm.intr.assume %cmp : i1
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ns = tt.splat %n : i32 -> tensor<64xi32>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %is = tt.splat %i : i32 -> tensor<64xi32>
        %idx = arith.addi %is, %lane : tensor<64xi32>
        %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
        scf.yield
      }
      tt.return
    })");
  auto p = proof(get("mask"));
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_FALSE(p.factsUsed.empty()); // provenance recorded
}

TEST_F(SymbolicBoundsTest, NoLoopQueryUsesConstantRanges) {
  parse(R"(
    tt.func @f(%p: i32) {
      %c0 = arith.constant 0 : i32
      %pid = tt.get_program_id x : i32
      %cmp = arith.cmpi sge, %pid, %c0 : i32 loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Satisfied"); // ctx.loop == null path
}

TEST_F(SymbolicBoundsTest, NoTimeoutPollBlocksBackwardAssume) {
  // tt.atomic_poll without a timeout polls until the value matches, so for
  // x < 0 it may spin forever and the assume after it never executes.
  parse(R"(
    tt.func @f(%p: !tt.ptr<i32>, %x: i32) {
      %c0 = arith.constant 0 : i32
      %cmp = arith.cmpi sge, %x, %c0 : i32 loc("cmp")
      %ok = tt.atomic_poll acquire, gpu, %p, %c0 : !tt.ptr<i32>, i32 -> i1
      %fact = arith.cmpi sge, %x, %c0 : i32
      llvm.intr.assume %fact : i1
      tt.return
    })");
  EXPECT_NE(verdict(get("cmp")), "Satisfied");
}

TEST_F(SymbolicBoundsTest, ImpureExternCallBlocksBackwardAssume) {
  // An impure tt.extern_elementwise lowers to an external call that may never
  // return, and it declares no CallOpInterface.
  parse(R"(
    tt.func @f(%x: i32) {
      %c0 = arith.constant 0 : i32
      %cmp = arith.cmpi sge, %x, %c0 : i32 loc("cmp")
      %e = tt.extern_elementwise %x {libname = "l", libpath = "p", symbol = "s", pure = false} : (i32) -> i32
      %fact = arith.cmpi sge, %x, %c0 : i32
      llvm.intr.assume %fact : i1
      tt.return
    })");
  EXPECT_NE(verdict(get("cmp")), "Satisfied");
}

TEST_F(SymbolicBoundsTest, QuotientFactIsNotDividendFact) {
  // assume(X % 2 == 0) is indexed under X, and q = X divsi 4 stores X as its
  // value. The exact loop end needs `q divisible by 2`, which that fact does
  // not give: at X = 12, q = 3, the loop runs i = 0, 2 and lane 1 of the last
  // iteration is false (3 < 3). Matching the fact by stored value would prove
  // Satisfied.
  parse(R"(
    tt.func @f(%X: i32) {
      %c0 = arith.constant 0 : i32
      %c2 = arith.constant 2 : i32
      %c4 = arith.constant 4 : i32
      %r = arith.remsi %X, %c2 : i32
      %eq = arith.cmpi eq, %r, %c0 : i32
      llvm.intr.assume %eq : i1
      %q = arith.divsi %X, %c4 : i32
      %lane = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
      %qs = tt.splat %q : i32 -> tensor<2xi32>
      scf.for %i = %c0 to %q step %c2 : i32 {
        %is = tt.splat %i : i32 -> tensor<2xi32>
        %idx = arith.addi %is, %lane : tensor<2xi32>
        %mask = arith.cmpi slt, %idx, %qs : tensor<2xi32> loc("mask")
        scf.yield
      }
      tt.return
    })");
  EXPECT_NE(verdict(get("mask")), "Satisfied");
}

//===----------------------------------------------------------------------===//
// Residual candidates: term sign, quotient threshold, residual guard
//===----------------------------------------------------------------------===//

TEST_F(SymbolicBoundsTest, ResidualSignGuardsProductValue) {
  parse(R"(
    tt.func @f(%a: i32, %b: i32) {
      %c0 = arith.constant 0 : i32
      %p = arith.muli %a, %b : i32 loc("p")
      %cmp = arith.cmpi sge, %p, %c0 : i32 loc("cmp")
      tt.return
    })");
  // p is Opaque (product of symbols). The guard is on p's own wrapped runtime
  // value, never on the factors: a = 65536, b = 32768 satisfy a >= 0 and b >= 0
  // while the i32 product is INT32_MIN.
  EXPECT_EQ(verdict(get("cmp")), "Conditional{opaque(p) >= 0}");
}

TEST_F(SymbolicBoundsTest, ResidualStrictlyPositive) {
  // The goal d >= 1 on an Opaque product needs p > 0, not p >= 0.
  parse(R"(
    tt.func @f(%a: i32, %b: i32) {
      %c0 = arith.constant 0 : i32
      %p = arith.muli %a, %b : i32 loc("p")
      %cmp = arith.cmpi sgt, %p, %c0 : i32 loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Conditional{opaque(p) > 0}");
}

TEST_F(SymbolicBoundsTest, QuotientThresholdPositive) {
  // q >= 2 translates to K >= 65 for the cdiv shape. It also exercises the
  // pruning dependency, since the retained K >= 65 may only drop a term-sign
  // `q >= 0` while the K + 63 wrap guard is retained.
  parse(R"(
    tt.func @f(%K: i32) {
      %c2 = arith.constant 2 : i32
      %c63 = arith.constant 63 : i32
      %c64 = arith.constant 64 : i32
      %num = arith.addi %K, %c63 : i32
      %q = arith.divsi %num, %c64 : i32
      %cmp = arith.cmpi sge, %q, %c2 : i32 loc("cmp")
      tt.return
    })");
  // `arg0 >= 0` is the division facts' precondition, but the retained fact
  // `arg0 >= 65` implies it outright, so closed-evidence pruning drops it; the
  // numerator wrap guard stays, since nothing implies it.
  EXPECT_EQ(verdict(get("cmp")), "Conditional{arg0 >= 65; arg0 <= 2147483584}");
}

TEST_F(SymbolicBoundsTest, QuotientThresholdUpperMirror) {
  // k < 0: 3 - q >= 1 needs q <= 2, i.e. (cdiv) K <= 128. The K + 63 wrap
  // guard is implied by K <= 128 and dropped; NonNegative(K) stays.
  parse(R"(
    tt.func @f(%K: i32) {
      %c3 = arith.constant 3 : i32
      %c63 = arith.constant 63 : i32
      %c64 = arith.constant 64 : i32
      %num = arith.addi %K, %c63 : i32
      %q = arith.divsi %num, %c64 : i32
      %cmp = arith.cmpi slt, %q, %c3 : i32 loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Conditional{arg0 <= 128; arg0 >= 0}");
}

TEST_F(SymbolicBoundsTest, MultiSymbolResidualGuard) {
  // d = hi - lo - 7 has two symbols, so no single-symbol candidate applies;
  // the residual guard covers the whole residual.
  parse(R"(
    tt.func @f(%lo: i32, %hi: i32) {
      %c0 = arith.constant 0 : i32
      %c8 = arith.constant 8 : i32
      %lane = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
      %d = arith.subi %hi, %lo : i32
      %ds = tt.splat %d : i32 -> tensor<8xi32>
      %mask = arith.cmpi slt, %lane, %ds : tensor<8xi32> loc("mask")
      tt.return
    })");
  // lo(d) = (hi - lo) - 7; the residual guard is AtLeast(hi - lo, 8). The
  // subi's wrap obligation must also be closed: lo = INT32_MIN, hi = 0
  // satisfies the mathematical hi - lo >= 8 while the narrow subtraction wraps
  // negative and the mask is false, so the upper representability guard is
  // not optional. Pin the conditions in full rather than by prefix.
  EXPECT_EQ(verdict(get("mask")),
            "Conditional{-arg0 + arg1 >= 8; -arg0 + arg1 <= 2147483647}");
}

TEST_F(SymbolicBoundsTest, CandidateProofStillClosesObligations) {
  parse(R"(
    tt.func @f(%x: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %y = arith.addi %x, %c1 : i32 loc("y")
      %cmp = arith.cmpi sge, %y, %c0 : i32 loc("cmp")
      tt.return
    })");
  // The term-sign candidate is the weaker x >= 0; the search also tries the
  // residual guard, whose exact AtLeast(y, 0) folds to the correct, wider
  // x >= -1 (y = x + 1 >= 0 admits x = -1, which x >= 0 wrongly excluded) and
  // wins by the width metric. finalize must still close the wrap obligation
  // on y with x <= INT32_MAX - 1 either way, otherwise x = INT32_MAX would
  // pass.
  EXPECT_EQ(verdict(get("cmp")), "Conditional{arg0 >= -1; arg0 <= 2147483646}");
}

TEST_F(SymbolicBoundsTest, MaterializeInsertsBeforeTheAnchor) {
  parse(R"(
    tt.func @f(%ptr: !tt.ptr<f32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ns = tt.splat %n : i32 -> tensor<64xi32>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %is = tt.splat %i : i32 -> tensor<64xi32>
        %idx = arith.addi %is, %lane : tensor<64xi32>
        %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
        scf.yield
      }
      tt.return
    })");
  tt::intel::BoundProof p = proof(get("mask"));
  // materialize returns a null Value for no conditions, so this must be
  // conditional for the placement to be observable.
  ASSERT_FALSE(p.conditions.empty());

  scf::ForOp loop;
  module->walk([&](scf::ForOp f) { loop = f; });
  Operation *term = func().getBody().front().getTerminator();

  // The builder deliberately points at the function terminator, not at the
  // anchor: the documented contract is that the guard goes immediately before
  // `before` whatever the caller's insertion point is.
  OpBuilder b(term);
  Value guard = tt::intel::materialize(p.conditions, loop, b);
  ASSERT_TRUE(guard);
  Operation *g = guard.getDefiningOp();

  // The guard is the last op emitted and sits right before the loop, and
  // nothing was emitted between the loop and the terminator.
  EXPECT_EQ(g->getNextNode(), loop.getOperation());
  EXPECT_EQ(loop->getNextNode(), term);
  // The caller's insertion point is restored.
  EXPECT_EQ(b.getInsertionBlock(), term->getBlock());
  EXPECT_TRUE(b.getInsertionPoint() == Block::iterator(term));
}

TEST_F(SymbolicBoundsTest, LoopRootedProverKeepsDistinctArgumentsDistinct) {
  const char *ir = R"(
    tt.func @f(%a: i32, %b: i32, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %n step %c1 : i32 {
        %cmp = arith.cmpi sge, %a, %b : i32 loc("cmp")
        scf.yield
      }
      tt.return
    })";
  parse(ir);
  std::string moduleRooted = verdict(get("cmp"));
  EXPECT_EQ(moduleRooted, "Conditional{arg0 - arg1 >= 0}");

  // A prover rooted at the loop numbers only what is under the loop; the two
  // function arguments are outside it. They used to tie in the sort order, so
  // `a - b` cancelled to 0 and `a >= b` read as Satisfied, which is false at
  // a = -1, b = 0.
  scf::ForOp loop;
  module->walk([&](scf::ForOp f) { loop = f; });
  prover = std::make_unique<tt::intel::SymbolicBoundsProver>(*solver, *domInfo,
                                                             loop);
  EXPECT_EQ(verdict(get("cmp")), moduleRooted);
  tt::intel::Symbol sa =
      prover->symbolFor(tt::intel::SymbolKind::KernelArg, arg(0));
  tt::intel::Symbol sb =
      prover->symbolFor(tt::intel::SymbolKind::KernelArg, arg(1));
  EXPECT_NE(sa.order(), sb.order());
}

TEST_F(SymbolicBoundsTest, ValuesCreatedAfterConstructionGetDistinctOrders) {
  parse(R"(
    tt.func @f(%a: i32) {
      tt.return
    })");

  // Neither value exists when the prover numbers the IR, so neither is in the
  // numbering. They used to share one fallback key, so their symbols tied.
  OpBuilder b(func().getBody().front().getTerminator());
  Location loc = b.getUnknownLoc();
  Value v1 = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 1);
  Value v2 = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 2);

  auto opaque = [&](Value v) {
    return prover->symbolFor(tt::intel::SymbolKind::Opaque, v);
  };
  unsigned o1 = opaque(v1).order(), o2 = opaque(v2).order();
  EXPECT_NE(o1, 0u);
  EXPECT_NE(o2, 0u);
  EXPECT_NE(o1, o2);
  // An order, once assigned, never changes.
  EXPECT_EQ(opaque(v1).order(), o1);
  EXPECT_EQ(opaque(v2).order(), o2);
  // And so v1 - v2 does not cancel.
  tt::intel::AffineForm diff =
      tt::intel::AffineForm::symbol(opaque(v1))
          .sub(tt::intel::AffineForm::symbol(opaque(v2)));
  EXPECT_EQ(diff.numTerms(), 2u);
}

TEST_F(SymbolicBoundsTest, MaskEvaluationsCountEachNodeOfASmallMask) {
  parse(R"(
    tt.func @f(%a: i32, %b: i32) {
      %c0 = arith.constant 0 : i32
      %x = arith.cmpi sge, %a, %c0 : i32
      %y = arith.cmpi sge, %b, %c0 : i32
      %m = arith.andi %x, %y : i1 loc("m")
      tt.return
    })");
  EXPECT_EQ(prover->numMaskEvaluations(), 0u);
  Value m = get("m");
  prover->proveTrue(m, at(m));
  // The conjunction and its two comparison operands.
  EXPECT_EQ(prover->numMaskEvaluations(), 3u);
}

//===----------------------------------------------------------------------===//
// proveTrue's walk over a mask expression: memo, depth cap, visit budget.
//===----------------------------------------------------------------------===//

using MaskProver = tt::intel::SymbolicBoundsProver;

/// IR text for a diamond ladder: every level feeds its input to two
/// conjunctions and joins them, so an unmemoized walk doubles per level, and
/// every route from the top down crosses two conjunctions per level. Needs a
/// `%x : i32` in scope. The top is `%<p>m<levels>`, named `topLoc` if given.
static std::string ladderOps(const std::string &p, unsigned levels,
                             const std::string &topLoc = "") {
  std::string s = "  %" + p + "z = arith.constant 0 : i32\n";
  s += "  %" + p + "m0 = arith.cmpi sge, %x, %" + p + "z : i32\n";
  for (unsigned k = 0; k < levels; ++k) {
    std::string i = std::to_string(k), n = std::to_string(k + 1);
    s += "  %" + p + "kc" + i + " = arith.constant " + n + " : i32\n";
    s += "  %" + p + "kd" + i + " = arith.constant -" + n + " : i32\n";
    s += "  %" + p + "c" + i + " = arith.cmpi sle, %x, %" + p + "kc" + i +
         " : i32\n";
    s += "  %" + p + "d" + i + " = arith.cmpi sge, %x, %" + p + "kd" + i +
         " : i32\n";
    s += "  %" + p + "p" + i + " = arith.andi %" + p + "m" + i + ", %" + p +
         "c" + i + " : i1\n";
    s += "  %" + p + "q" + i + " = arith.andi %" + p + "m" + i + ", %" + p +
         "d" + i + " : i1\n";
    s += "  %" + p + "m" + n + " = arith.andi %" + p + "p" + i + ", %" + p +
         "q" + i + " : i1";
    if (k + 1 == levels && !topLoc.empty())
      s += " loc(\"" + topLoc + "\")";
    s += "\n";
  }
  return s;
}

/// IR text for a chain of `n` conjunctions over constant true, bottom
/// `%<p>0`, top `%<p><n>`, named `topLoc` if given. Needs `%t` in scope. Every
/// node is distinct, so a memo does not shorten it.
static std::string chainOps(const std::string &p, unsigned n,
                            const std::string &topLoc = "") {
  std::string s = "  %" + p + "0 = arith.constant true\n";
  for (unsigned k = 1; k <= n; ++k) {
    s += "  %" + p + std::to_string(k) + " = arith.andi %" + p +
         std::to_string(k - 1) + ", %t : i1";
    if (k == n && !topLoc.empty())
      s += " loc(\"" + topLoc + "\")";
    s += "\n";
  }
  return s;
}

static std::string funcOf(const std::string &args, const std::string &ops) {
  return "tt.func @f(" + args + ") {\n" + ops + "  tt.return\n}\n";
}

/// A comparison of constants that is false, so `proveTrue` refutes it.
static const char *kRefutedOps = "  %five = arith.constant 5 : i32\n"
                                 "  %three = arith.constant 3 : i32\n"
                                 "  %r = arith.cmpi slt, %five, %three : i32\n";

TEST_F(SymbolicBoundsTest, MaskMemoMakesASharedLadderLinear) {
  // Two conjunctions per level, so the depth stays well under kMaxMaskDepth
  // and only the memo can help.
  constexpr unsigned levels = 12;
  parse(funcOf("%x: i32", ladderOps("", levels, "top")));
  Value top = get("top");
  prover->proveTrue(top, at(top));
  // Per level two comparisons and three conjunctions, plus the first
  // comparison: each node evaluated once.
  EXPECT_LE(prover->numMaskEvaluations(), 5 * levels + 1);
}

TEST_F(SymbolicBoundsTest, MaskDepthCapTurnsALongChainUnknown) {
  auto verdictOfChain = [&](unsigned n) {
    parse(funcOf("", "  %t = arith.constant true\n" + chainOps("c", n, "top")));
    Value top = get("top");
    return tt::intel::toString(prover->proveTrue(top, at(top)));
  };
  EXPECT_EQ(verdictOfChain(MaskProver::kMaxMaskDepth), "Satisfied");
  EXPECT_EQ(verdictOfChain(MaskProver::kMaxMaskDepth + 1), "Unknown");
}

// A subtree of height 40 reached near the top is within the cap; reached below
// a 40-deep chain it is not. Both uses share the subtree.
static std::string sharedSubtreeIR() {
  return funcOf(
      "", "  %t = arith.constant true\n" + chainOps("s", 40) +
              "  %shallow = arith.andi %s40, %t : i1 loc(\"shallow\")\n"
              "  %e1 = arith.andi %s40, %t : i1\n" +
              [] {
                std::string d;
                for (unsigned k = 2; k <= 40; ++k)
                  d += "  %e" + std::to_string(k) + " = arith.andi %e" +
                       std::to_string(k - 1) + ", %t : i1" +
                       (k == 40 ? " loc(\"deep\")" : "") + "\n";
                return d;
              }());
}

TEST_F(SymbolicBoundsTest, MaskReuseShallowThenDeep) {
  parse(sharedSubtreeIR());
  Value shallow = get("shallow"), deep = get("deep");
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(shallow, at(shallow))),
            "Satisfied");
  // The cached subtree is within the cap where it was computed; reached from
  // here it is not, and the cap must still apply.
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(deep, at(deep))), "Unknown");
}

TEST_F(SymbolicBoundsTest, MaskReuseDeepThenShallow) {
  parse(sharedSubtreeIR());
  Value shallow = get("shallow"), deep = get("deep");
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(deep, at(deep))), "Unknown");
  // The deep failure was a truncation, not an answer about the subtree, so it
  // must not make the shallow query fail.
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(shallow, at(shallow))),
            "Satisfied");
}

TEST_F(SymbolicBoundsTest, MaskDepthTruncationStillLetsARefutedSiblingRefute) {
  // Preservation: a truncated operand is Unknown, and a refuted operand wins.
  parse(funcOf("", "  %t = arith.constant true\n" + std::string(kRefutedOps) +
                       chainOps("c", MaskProver::kMaxMaskDepth + 5) +
                       "  %rt = arith.andi %r, %c" +
                       std::to_string(MaskProver::kMaxMaskDepth + 5) +
                       " : i1 loc(\"rt\")\n"
                       "  %tr = arith.andi %c" +
                       std::to_string(MaskProver::kMaxMaskDepth + 5) +
                       ", %r : i1 loc(\"tr\")\n"));
  for (const char *name : {"rt", "tr"}) {
    Value v = get(name);
    unsigned before = prover->numMaskEvaluations();
    EXPECT_EQ(tt::intel::toString(prover->proveTrue(v, at(v))), "Refuted")
        << name;
    unsigned first = prover->numMaskEvaluations() - before;
    // The result depended on a truncated branch, so it was not cached as a
    // complete answer: asking again evaluates again.
    prover->proveTrue(v, at(v));
    EXPECT_GT(prover->numMaskEvaluations() - before, first) << name;
  }
}

TEST_F(SymbolicBoundsTest, MaskVisitBudgetBoundsAnOverCapSharedLadder) {
  // 40 levels put every route 80 conjunctions deep, past the cap, so nothing
  // above the comparisons is ever complete enough to cache and the walk would
  // double per level. Only the visit budget stops it.
  parse(funcOf("%x: i32", ladderOps("", 40, "top")));
  Value top = get("top");
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(top, at(top))), "Unknown");
  EXPECT_LE(prover->numMaskEvaluations(), MaskProver::kMaxMaskVisits);
}

TEST_F(SymbolicBoundsTest, MaskBudgetExhaustionOverridesARefutedSibling) {
  // Budget exhaustion is query-wide: a refutation found before or after it
  // does not rescue the answer. Depth truncation, by contrast, is local.
  parse(
      funcOf("%x: i32", std::string(kRefutedOps) + ladderOps("", 40) +
                            "  %rt = arith.andi %r, %m40 : i1 loc(\"rt\")\n"
                            "  %tr = arith.andi %m40, %r : i1 loc(\"tr\")\n"));
  for (const char *name : {"rt", "tr"}) {
    Value v = get(name);
    EXPECT_EQ(tt::intel::toString(prover->proveTrue(v, at(v))), "Unknown")
        << name;
  }
}

TEST_F(SymbolicBoundsTest, MaskEntriesCachedBeforeExhaustionAreReused) {
  parse(funcOf("%x: i32",
               "  %t = arith.constant true\n" + chainOps("s", 10) +
                   "  %small = arith.andi %s10, %t : i1 loc(\"small\")\n" +
                   ladderOps("", 40, "big")));
  Value small = get("small"), big = get("big");
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(small, at(small))),
            "Satisfied");
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(big, at(big))), "Unknown");
  unsigned before = prover->numMaskEvaluations();
  // The complete answer computed before the budget ran out is still valid and
  // is served from the memo without evaluating anything.
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(small, at(small))),
            "Satisfied");
  EXPECT_EQ(prover->numMaskEvaluations(), before);
}

TEST_F(SymbolicBoundsTest, MaskMemoKeepsPointsAndLoopsApart) {
  // Preservation: a cached answer must not leak between query points or loop
  // contexts. For each pair, the answer on a shared prover equals the answer
  // on a fresh one, in either order.
  auto check = [&](const char *ir, auto makeContexts) {
    auto fresh = [&](unsigned which) {
      parse(ir);
      auto [v, a, b] = makeContexts();
      return tt::intel::toString(prover->proveTrue(v, which == 0 ? a : b));
    };
    std::string wantA = fresh(0), wantB = fresh(1);
    ASSERT_NE(wantA, wantB) << "the two contexts must disagree";
    for (bool aFirst : {true, false}) {
      parse(ir);
      auto [v, a, b] = makeContexts();
      std::string gotA, gotB;
      if (aFirst) {
        gotA = tt::intel::toString(prover->proveTrue(v, a));
        gotB = tt::intel::toString(prover->proveTrue(v, b));
      } else {
        gotB = tt::intel::toString(prover->proveTrue(v, b));
        gotA = tt::intel::toString(prover->proveTrue(v, a));
      }
      EXPECT_EQ(gotA, wantA) << (aFirst ? "A then B" : "B then A");
      EXPECT_EQ(gotB, wantB) << (aFirst ? "A then B" : "B then A");
    }
  };

  // Two program points: an assume that holds at the later one only, because a
  // region-bearing op between the earlier point and the assume stops it from
  // applying backward.
  check(R"(
    tt.func @f(%n: i32, %flag: i1) {
      %c0 = arith.constant 0 : i32
      %m = arith.cmpi sge, %n, %c0 : i32 loc("m")
      %early = arith.constant 1 : i32 loc("early")
      scf.if %flag {
      }
      %ge = arith.cmpi sge, %n, %c0 : i32
      llvm.intr.assume %ge : i1
      %late = arith.constant 2 : i32 loc("late")
      tt.return
    })",
        [&] {
          auto opOf = [&](const char *n) { return get(n).getDefiningOp(); };
          return std::make_tuple(
              get("m"), tt::intel::QueryContext{opOf("early"), nullptr},
              tt::intel::QueryContext{opOf("late"), nullptr});
        });

  // Two loop contexts for one value: inside its loop, and with no loop.
  check(R"(
    tt.func @f(%ptr: !tt.ptr<f32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c64 = arith.constant 64 : i32
      %r = arith.remsi %n, %c64 : i32
      %cmp = arith.cmpi eq, %r, %c0 : i32
      llvm.intr.assume %cmp : i1
      %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %ns = tt.splat %n : i32 -> tensor<64xi32>
      scf.for %i = %c0 to %n step %c64 : i32 {
        %is = tt.splat %i : i32 -> tensor<64xi32>
        %idx = arith.addi %is, %lane : tensor<64xi32>
        %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32> loc("mask")
        scf.yield
      }
      tt.return
    })",
        [&] {
          Value mask = get("mask");
          tt::intel::QueryContext inLoop = at(mask);
          tt::intel::QueryContext noLoop{inLoop.at, nullptr};
          return std::make_tuple(mask, inLoop, noLoop);
        });
}

TEST_F(SymbolicBoundsTest, MaskQueriesRecoverAfterTruncation) {
  // Preservation: a query cut short by the depth cap leaves the prover able to
  // decide an ordinary one.
  parse(funcOf("", "  %t = arith.constant true\n" +
                       chainOps("c", MaskProver::kMaxMaskDepth + 5, "long") +
                       "  %ok = arith.andi %t, %t : i1 loc(\"ok\")\n"));
  Value longMask = get("long"), ok = get("ok");
  prover->proveTrue(longMask, at(longMask));
  EXPECT_EQ(tt::intel::toString(prover->proveTrue(ok, at(ok))), "Satisfied");
}

//===----------------------------------------------------------------------===//
// factsUsed: every assume-derived range a proof used is reported.
//===----------------------------------------------------------------------===//

/// The `llvm.intr.assume` operations under `module`, in program order.
static SmallVector<Operation *> assumesOf(ModuleOp module) {
  SmallVector<Operation *> out;
  module.walk([&](LLVM::AssumeOp a) { out.push_back(a); });
  return out;
}

/// `proof` consulted every one of `wanted`.
static ::testing::AssertionResult usesAll(const tt::intel::BoundProof &proof,
                                          ArrayRef<Operation *> wanted) {
  for (Operation *a : wanted)
    if (!llvm::is_contained(proof.factsUsed, a)) {
      std::string where;
      llvm::raw_string_ostream os(where);
      a->getLoc().print(os);
      return ::testing::AssertionFailure()
             << "missing the assume at " << where << "; factsUsed has "
             << proof.factsUsed.size();
    }
  return ::testing::AssertionSuccess();
}

TEST_F(SymbolicBoundsTest, FactsUsedRecordsAssumesBehindAWrapDischarge) {
  const char *withAssumes = R"(
    tt.func @f(%a: i8) {
      %c1 = arith.constant 1 : i8
      %c100 = arith.constant 100 : i8
      %c110 = arith.constant 110 : i8
      %ge = arith.cmpi sge, %a, %c100 : i8
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %a, %c110 : i8
      llvm.intr.assume %le : i1
      %y = arith.addi %a, %c1 : i8
      %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
      tt.return
    })";
  parse(withAssumes);
  tt::intel::BoundProof p = proof(get("cmp"));
  // y = a + 1 cannot wrap for a in [100, 110]; only the assumes say so.
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_TRUE(usesAll(p, assumesOf(module.get())));

  // Control: without the assumes the same query needs a runtime condition.
  parse(R"(
    tt.func @f(%a: i8) {
      %c1 = arith.constant 1 : i8
      %y = arith.addi %a, %c1 : i8
      %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
      tt.return
    })");
  EXPECT_EQ(verdict(get("cmp")), "Conditional{arg0 <= 126}");
}

TEST_F(SymbolicBoundsTest, FactsUsedRecordsAssumesBehindAPrunedCondition) {
  parse(R"(
    tt.func @f(%n: i32) {
      %c40 = arith.constant 40 : i32
      %c50 = arith.constant 50 : i32
      %le = arith.cmpi sle, %n, %c40 : i32
      llvm.intr.assume %le : i1
      %cmp = arith.cmpi slt, %n, %c50 : i32 loc("cmp")
      tt.return
    })");
  tt::intel::BoundProof p = proof(get("cmp"));
  // n <= 40 makes the runtime condition n <= 49 redundant, so it is pruned.
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_TRUE(usesAll(p, assumesOf(module.get())));
}

// A loop-varying value that normalization leaves opaque and whose own range
// inherits a function argument's assume-narrowed range. (A loaded value does
// not do: it gets its entry state, and its assumes narrow only its uses.)
static const char *kVaryingRangeIR = R"(
    tt.func @f(%arg: i32, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c10 = arith.constant 10 : i32
      %c50 = arith.constant 50 : i32
      %c100 = arith.constant 100 : i32
      %ge = arith.cmpi sge, %arg, %c0 : i32
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %arg, %c10 : i32
      llvm.intr.assume %le : i1
      %t = arith.constant true
      scf.for %i = %c0 to %n step %c1 : i32 {
        %y = arith.minsi %arg, %c100 : i32
        %gt = arith.cmpi sgt, %y, %c50 : i32 loc("gt")
        %lt = arith.cmpi slt, %y, %c50 : i32 loc("lt")
        %gt_and_t = arith.andi %gt, %t : i1 loc("gt_and_t")
        %t_and_gt = arith.andi %t, %gt : i1 loc("t_and_gt")
        scf.yield
      }
      tt.return
    })";

TEST_F(SymbolicBoundsTest, FactsUsedRecordsAssumesBehindAVaryingRange) {
  parse(kVaryingRangeIR);
  SmallVector<Operation *> assumes = assumesOf(module.get());
  tt::intel::BoundProof gt = proof(get("gt")), lt = proof(get("lt"));
  EXPECT_EQ(tt::intel::toString(gt), "Refuted");
  EXPECT_EQ(tt::intel::toString(lt), "Satisfied");
  EXPECT_TRUE(usesAll(gt, assumes));
  EXPECT_TRUE(usesAll(lt, assumes));
}

TEST_F(SymbolicBoundsTest, FactsUsedSurvivesAConjunctionThatRefutes) {
  parse(kVaryingRangeIR);
  SmallVector<Operation *> assumes = assumesOf(module.get());
  // The refuted operand in either position.
  for (const char *name : {"gt_and_t", "t_and_gt"}) {
    Value v = get(name);
    tt::intel::BoundProof p = prover->proveTrue(v, at(v));
    EXPECT_EQ(tt::intel::toString(p), "Refuted") << name;
    EXPECT_TRUE(usesAll(p, assumes)) << name;
  }
}

// The wrap discharge of `a + 1 > a` that only holds because of the assumes on
// `%a`, queried with no program point.
static const char *kUpstreamAssumeLoopIR = R"(
    tt.func @f(%a: i8, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %k1 = arith.constant 1 : i8
      %c100 = arith.constant 100 : i8
      %c110 = arith.constant 110 : i8
      %ge = arith.cmpi sge, %a, %c100 : i8
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %a, %c110 : i8
      llvm.intr.assume %le : i1
      scf.for %i = %c0 to %n step %c1 : i32 {
        %y = arith.addi %a, %k1 : i8
        %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
        scf.yield
      }
      tt.return
    })";

/// Proves the comparison named `cmp` with no program point and no loop.
static tt::intel::BoundProof
proveWithoutContext(tt::intel::SymbolicBoundsProver &prover, Value cmp) {
  auto op = cast<arith::CmpIOp>(cmp.getDefiningOp());
  return prover.prove(op.getPredicate(), op.getLhs(), op.getRhs(),
                      tt::intel::QueryContext{nullptr, nullptr});
}

TEST_F(SymbolicBoundsTest, FactsUsedWithNoPointReachesAssumesOutsideTheRoot) {
  parse(kUpstreamAssumeLoopIR);
  SmallVector<Operation *> assumes = assumesOf(module.get());
  // Rooted at the loop, the assumes before it are outside the root; the range
  // of `%a` still comes from them.
  scf::ForOp loop;
  module->walk([&](scf::ForOp f) { loop = f; });
  prover = std::make_unique<tt::intel::SymbolicBoundsProver>(*solver, *domInfo,
                                                             loop);
  tt::intel::BoundProof p = proveWithoutContext(*prover, get("cmp"));
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_TRUE(usesAll(p, assumes));
}

TEST_F(SymbolicBoundsTest, FactsUsedWithNoPointComesFromTheValuesOwnFunction) {
  parse(R"(
    tt.func @other(%p: i8, %n: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c10 = arith.constant 10 : i8
      %le = arith.cmpi sle, %p, %c10 : i8
      llvm.intr.assume %le : i1
      scf.for %i = %c0 to %n step %c1 : i32 {
        scf.yield
      }
      tt.return
    }
    tt.func @f(%a: i8) {
      %k1 = arith.constant 1 : i8
      %c100 = arith.constant 100 : i8
      %c110 = arith.constant 110 : i8
      %ge = arith.cmpi sge, %a, %c100 : i8
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %a, %c110 : i8
      llvm.intr.assume %le : i1
      %y = arith.addi %a, %k1 : i8
      %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
      tt.return
    })");
  SmallVector<Operation *> assumes = assumesOf(module.get());
  ASSERT_EQ(assumes.size(), 3u);
  Operation *otherAssume = assumes[0];
  // Rooted in the other function; the query is in this one.
  scf::ForOp loop;
  module->walk([&](scf::ForOp f) { loop = f; });
  prover = std::make_unique<tt::intel::SymbolicBoundsProver>(*solver, *domInfo,
                                                             loop);
  tt::intel::BoundProof p = proveWithoutContext(*prover, get("cmp"));
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_TRUE(usesAll(p, {assumes[1], assumes[2]}));
  EXPECT_FALSE(llvm::is_contained(p.factsUsed, otherAssume));
}

TEST_F(SymbolicBoundsTest, FactsUsedReachesTheCallersAssumesAcrossACall) {
  // The range analysis is interprocedural: a private callee's argument takes
  // the range of what its call sites pass, so the caller's assumes narrow it.
  parse(R"(
    tt.func private @callee(%a: i8) -> i1 {
      %c1 = arith.constant 1 : i8
      %y = arith.addi %a, %c1 : i8
      %cmp = arith.cmpi sgt, %y, %a : i8 loc("cmp")
      tt.return %cmp : i1
    }
    tt.func public @caller(%x: i8) {
      %c0 = arith.constant 0 : i8
      %c10 = arith.constant 10 : i8
      %ge = arith.cmpi sge, %x, %c0 : i8
      llvm.intr.assume %ge : i1
      %le = arith.cmpi sle, %x, %c10 : i8
      llvm.intr.assume %le : i1
      %r = tt.call @callee(%x) : (i8) -> i1
      tt.return
    })");
  tt::intel::BoundProof p = proveWithoutContext(*prover, get("cmp"));
  EXPECT_EQ(tt::intel::toString(p), "Satisfied");
  EXPECT_TRUE(usesAll(p, assumesOf(module.get())));
}

} // namespace
