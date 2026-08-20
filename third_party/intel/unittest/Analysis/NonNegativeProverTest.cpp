//===- NonNegativeProverTest.cpp ------------------------------------------===//
//
// Unit tests for triton::intel::proveSign(), the goal-directed sign prover
// behind TritonIntelSpeculateSignedDivRem.
//
// Every decomposition rule is a soundness claim ("the result meets the goal if
// these operands do"), and a wrong one lets the pass convert an operation whose
// dividend can be negative. A lit test can only observe the enclosing pass's
// aggregate verdict, so each rule is exercised here directly, in both
// directions: the fact it establishes, and the fact it declines to establish.
//
// The cases are written so the range analysis cannot answer the top-level
// question by itself - typically by rooting a chain at a function argument,
// whose range is unbounded. Otherwise the prover would answer from the range
// analysis alone and the rule under test would never run.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/NonNegativeProver.h"
#include "intel/include/Analysis/Range.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

using namespace mlir;
namespace tt = mlir::triton;

using tt::intel::Goal;
using tt::intel::Obligation;
using tt::intel::Proof;

namespace {

class NonNegativeProverTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<tt::TritonDialect>();
  }

  /// Parses `ir` and runs the range analysis over it, exactly as the pass does
  /// (see SpeculateSignedDivRem.cpp). Must be called before prove().
  void parse(StringRef ir) {
    module = parseSourceString<ModuleOp>(ir, &ctx);
    ASSERT_TRUE(module) << "failed to parse:\n" << ir.str();
    ModuleOp mod = module.get();
    domInfo = std::make_unique<DominanceInfo>(mod);
    solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(mod, *domInfo);
    ASSERT_TRUE(succeeded(solver->initializeAndRun(mod)));
  }

  Proof prove(Value v, Goal goal) {
    return tt::intel::proveSign(v, goal, *solver);
  }

  /// Returns the single result of the operation carrying `loc("<name>")` in the
  /// parsed IR. Naming the interesting operations by location keeps the tests
  /// independent of the SSA numbering the parser assigns.
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

  /// Renders a proof as a stable, readable string, so a failure reports what
  /// was proven rather than a pointer comparison. Obligations are named after
  /// the `loc("...")` of the operation that defines them.
  ///
  /// `Satisfied` and `Refuted` promise an empty obligation list. One that
  /// arrives non-empty anyway - an abandoned sub-proof leaking its obligations
  /// past the verdict that replaced it - renders as `Refuted+STRAY{...}` rather
  /// than dropping them, so the comparison fails instead of matching plain
  /// `"Refuted"` and passing on a proof that broke its own contract.
  std::string render(const Proof &proof) {
    std::string out;
    llvm::raw_string_ostream os(out);

    switch (proof.verdict) {
    case Proof::Satisfied:
      os << "Satisfied";
      break;
    case Proof::Refuted:
      os << "Refuted";
      break;
    case Proof::Obligated:
      os << "Obligated";
      break;
    }

    if (proof.verdict != Proof::Obligated) {
      if (proof.obligations.empty())
        return out;
      os << "+STRAY";
    }

    os << "{";
    llvm::interleaveComma(proof.obligations, os, [&](const Obligation &o) {
      os << name(o.first) << (o.second == Goal::NonNegative ? " >= 0" : " > 0");
    });
    os << "}";
    return out;
  }

private:
  /// A value's `loc("...")` name, or `<op>#<n>` for a block argument.
  static std::string name(Value v) {
    if (Operation *defOp = v.getDefiningOp()) {
      if (auto nameLoc = dyn_cast<NameLoc>(defOp->getLoc()))
        return nameLoc.getName().str();
      return defOp->getName().getStringRef().str();
    }
    auto blockArg = cast<BlockArgument>(v);
    Operation *owner = blockArg.getOwner()->getParentOp();
    return (owner->getName().getStringRef() + "#" +
            Twine(blockArg.getArgNumber()))
        .str();
  }

  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<DataFlowSolver> solver;
};

//===----------------------------------------------------------------------===//
// The range analysis answers, and no rule runs
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, RangeAnalysisAnswersSatisfied) {
  parse(R"mlir(
    tt.func @f() {
      %c8 = arith.constant 8 : i32 loc("pos_const")
      %pid = tt.get_program_id x : i32 loc("pid")
      %r = tt.make_range {start = 0 : i32, end = 256 : i32}
           : tensor<256xi32> loc("range")
      %c64 = arith.constant 64 : i32
      %m = arith.remui %pid, %c64 : i32 loc("remui")
      tt.return
    }
  )mlir");

  // Each of these is decided by the range analysis alone. That the prover
  // consults it first is what keeps it from duplicating it.
  EXPECT_EQ(render(prove(get("pos_const"), Goal::NonNegative)), "Satisfied");
  EXPECT_EQ(render(prove(get("pos_const"), Goal::Positive)), "Satisfied");
  EXPECT_EQ(render(prove(get("pid"), Goal::NonNegative)), "Satisfied");
  EXPECT_EQ(render(prove(get("range"), Goal::NonNegative)), "Satisfied");
  EXPECT_EQ(render(prove(get("remui"), Goal::NonNegative)), "Satisfied");

  // tt.get_program_id is [0, INT_MAX-1] and tt.make_range starts at 0, so
  // neither is provably positive. But neither is provably non-positive either,
  // so the verdict is an obligation, not a refutation - "not proven" and
  // "disproven" are distinct, and only the latter is a reason to stop looking.
  //
  // Note for the consumer: these two obligations are satisfiable in principle
  // yet violated on every launch, since program 0 always exists. Accepting an
  // obligation therefore cannot be a question of provenance alone.
  EXPECT_EQ(render(prove(get("pid"), Goal::Positive)), "Obligated{pid > 0}");
  EXPECT_EQ(render(prove(get("range"), Goal::Positive)),
            "Obligated{range > 0}");
}

TEST_F(NonNegativeProverTest, RangeAnalysisAnswersRefuted) {
  parse(R"mlir(
    tt.func @f() {
      %cn8 = arith.constant -8 : i32 loc("neg_const")
      %c0 = arith.constant 0 : i32 loc("zero")
      %c2 = arith.constant 2 : i32
      %m = arith.muli %cn8, %c2 : i32 loc("neg_product")
      tt.return
    }
  )mlir");

  EXPECT_EQ(render(prove(get("neg_const"), Goal::NonNegative)), "Refuted");
  EXPECT_EQ(render(prove(get("neg_product"), Goal::NonNegative)), "Refuted");

  // Zero satisfies NonNegative but refutes Positive. This is the distinction
  // that makes the emitted check `sgt` rather than `sge`: a divisor range of
  // [0, 4] is not enough for MLIR's inferDivS.
  EXPECT_EQ(render(prove(get("zero"), Goal::NonNegative)), "Satisfied");
  EXPECT_EQ(render(prove(get("zero"), Goal::Positive)), "Refuted");
}

//===----------------------------------------------------------------------===//
// Conjunctive rules: addi, muli, minsi, select
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, ConjunctiveRules) {
  parse(R"mlir(
    tt.func @f(%arg0: i32, %arg1: i32, %cond: i1) {
      %c8 = arith.constant 8 : i32
      %cn8 = arith.constant -8 : i32
      %a1 = arith.addi %arg0, %c8 : i32 loc("add_one_unknown")
      %a2 = arith.addi %arg0, %arg1 : i32 loc("add_two_unknown")
      %a3 = arith.addi %arg0, %cn8 : i32 loc("add_refuted")
      %m1 = arith.muli %arg0, %c8 : i32 loc("mul_one_unknown")
      %n1 = arith.minsi %arg0, %c8 : i32 loc("min_one_unknown")
      %s1 = arith.select %cond, %arg0, %c8 : i32 loc("select_one_unknown")
      tt.return
    }
  )mlir");

  // One operand known, the other not: a single obligation, on the operand the
  // recursion could not discharge.
  EXPECT_EQ(render(prove(get("add_one_unknown"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("mul_one_unknown"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("select_one_unknown"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");

  // Both unknown: the conjunction is the union.
  EXPECT_EQ(render(prove(get("add_two_unknown"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0, tt.func#1 >= 0}");

  // A refuted operand abandons the whole proof - no runtime check on the other
  // operand could rescue it.
  EXPECT_EQ(render(prove(get("add_refuted"), Goal::NonNegative)), "Refuted");

  // The goal is carried unchanged into both operands, so Positive on a sum
  // demands Positive of each. Pessimistic (1 + -0 is not the only way to be
  // positive) but sound.
  EXPECT_EQ(render(prove(get("add_one_unknown"), Goal::Positive)),
            "Obligated{tt.func#0 > 0}");

  // min(a, b) meets a goal exactly when both operands do, for either goal.
  EXPECT_EQ(render(prove(get("min_one_unknown"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("min_one_unknown"), Goal::Positive)),
            "Obligated{tt.func#0 > 0}");
}

//===----------------------------------------------------------------------===//
// maxsi: the one disjunctive rule
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, MaxSIDiscardsLosingBranch) {
  parse(R"mlir(
    tt.func @f(%arg0: i32, %arg1: i32) {
      %m = arith.maxsi %arg0, %arg1 : i32 loc("max_both_unknown")
      tt.return
    }
  )mlir");

  // Neither branch is satisfied unconditionally, and `a >= 0 or b >= 0` is not
  // a conjunction of facts a runtime check can express. So both sub-proofs are
  // discarded - one obligation, on the max itself, not two on the operands.
  EXPECT_EQ(render(prove(get("max_both_unknown"), Goal::NonNegative)),
            "Obligated{max_both_unknown >= 0}");
}

TEST_F(NonNegativeProverTest, MaxSISatisfiedBranchDischarges) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c64 = arith.constant 64 : i32
      %pid = tt.get_program_id x : i32
      %nn = arith.remui %pid, %c64 : i32
      %r = arith.remsi %nn, %arg0 : i32
      %m = arith.maxsi %r, %arg0 : i32 loc("max_one_satisfied")
      tt.return
    }
  )mlir");

  // %r is satisfied by the remsi rule, not by the range analysis, which does
  // not know a signed remainder takes the sign of its dividend. So the branch
  // that discharges the max here can only come from the recursion.
  EXPECT_EQ(render(prove(get("max_one_satisfied"), Goal::NonNegative)),
            "Satisfied");
}

//===----------------------------------------------------------------------===//
// Division and remainder: the load-bearing asymmetry
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, DivSIObligatesTheDivisorNotTheDividend) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c64 = arith.constant 64 : i32
      %pid = tt.get_program_id x : i32
      %nn = arith.remui %pid, %c64 : i32
      %d = arith.divsi %nn, %arg0 : i32 loc("div")
      %fd = arith.floordivsi %nn, %arg0 : i32 loc("floordiv")
      %cd = arith.ceildivsi %nn, %arg0 : i32 loc("ceildiv")
      tt.return
    }
  )mlir");

  // NN(divsi a, b) = NN(a) and P(b). NN(a) is the same question one step back,
  // discharged by continuing; P(b) is a different question about a different
  // value, and it is where the obligation lands. Note the goal is Positive:
  // divui reinterprets a negative divisor, and a zero divisor traps.
  for (StringRef op : {"div", "floordiv", "ceildiv"})
    EXPECT_EQ(render(prove(get(op), Goal::NonNegative)),
              "Obligated{tt.func#0 > 0}")
        << "for " << op.str();

  // No rule establishes strict positivity of a quotient: 1 / 2 is 0.
  EXPECT_EQ(render(prove(get("div"), Goal::Positive)), "Obligated{div > 0}");
}

TEST_F(NonNegativeProverTest, DivSIObligatesTheDividendWhenItIsTheBlocker) {
  parse(R"mlir(
    tt.func @f(%arg0: !tt.ptr<i32>) {
      %c4 = arith.constant 4 : i32
      %l = tt.load %arg0 : !tt.ptr<i32> loc("load")
      %d = arith.divsi %l, %c4 : i32 loc("div")
      tt.return
    }
  )mlir");

  // The counter-case to the test above: with a constant positive divisor the
  // dividend is the only thing left to prove, so the obligation lands on the
  // load. Reporting it is not the same as agreeing to assert it - declining a
  // load-derived obligation is the caller's policy, not the prover's.
  EXPECT_EQ(render(prove(get("div"), Goal::NonNegative)),
            "Obligated{load >= 0}");
}

TEST_F(NonNegativeProverTest, RemSIFollowsItsDividend) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c64 = arith.constant 64 : i32
      %c4 = arith.constant 4 : i32
      %pid = tt.get_program_id x : i32
      %nn = arith.remui %pid, %c64 : i32
      %r1 = arith.remsi %nn, %arg0 : i32 loc("rem_known_dividend")
      %r2 = arith.remsi %arg0, %c4 : i32 loc("rem_unknown_dividend")
      tt.return
    }
  )mlir");

  // A signed remainder takes the sign of its dividend, so the divisor is
  // irrelevant to non-negativity - no obligation on %arg0 here. This rule is
  // sharper than MLIR's inferRemS, which cannot prove this (verified with
  // -triton-intel-fold-true-cmpi as an oracle), so it is a strict improvement
  // over the range analysis rather than a restatement of it.
  EXPECT_EQ(render(prove(get("rem_known_dividend"), Goal::NonNegative)),
            "Satisfied");

  EXPECT_EQ(render(prove(get("rem_unknown_dividend"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");

  // The remainder can be zero, so nothing is claimed about positivity.
  EXPECT_EQ(render(prove(get("rem_known_dividend"), Goal::Positive)),
            "Obligated{rem_known_dividend > 0}");
}

//===----------------------------------------------------------------------===//
// Sign-preserving and shape operations
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, SignPreservingOps) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c2 = arith.constant 2 : i32
      %e = arith.extsi %arg0 : i32 to i64 loc("extsi")
      %s = arith.shrsi %arg0, %c2 : i32 loc("shrsi")
      %t = arith.trunci %arg0 : i32 to i16 loc("trunci")
      tt.return
    }
  )mlir");

  // Sign extension preserves the value, so it preserves both goals.
  EXPECT_EQ(render(prove(get("extsi"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("extsi"), Goal::Positive)),
            "Obligated{tt.func#0 > 0}");

  // An arithmetic shift right preserves the sign bit, but shifts a positive
  // value down to zero, so only the non-negative goal propagates.
  EXPECT_EQ(render(prove(get("shrsi"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("shrsi"), Goal::Positive)),
            "Obligated{shrsi > 0}");

  // Truncation can set the high bit of the narrower type, so it propagates
  // nothing and the obligation stays on the truncated value.
  EXPECT_EQ(render(prove(get("trunci"), Goal::NonNegative)),
            "Obligated{trunci >= 0}");
}

TEST_F(NonNegativeProverTest, ShapeOpsReduceToTheirScalarSource) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %s = tt.splat %arg0 : i32 -> tensor<16xi32>
      %e = tt.expand_dims %s {axis = 0 : i32}
           : tensor<16xi32> -> tensor<1x16xi32>
      %b = tt.broadcast %e : tensor<1x16xi32> -> tensor<8x16xi32> loc("bcast")
      tt.return
    }
  )mlir");

  // A goal on a tensor is a goal on every element, so a replicating operation
  // reduces to its source. The obligation is therefore a scalar even though
  // the question was about a tensor - which is what makes the runtime check
  // affordable.
  EXPECT_EQ(render(prove(get("bcast"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
  EXPECT_EQ(render(prove(get("bcast"), Goal::Positive)),
            "Obligated{tt.func#0 > 0}");
}

//===----------------------------------------------------------------------===//
// Where the recursion stops
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, SubIPlants) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c16 = arith.constant 16 : i32
      %s = arith.subi %c16, %arg0 : i32 loc("subi")
      tt.return
    }
  )mlir");

  // `a >= 0` and `b >= 0` say nothing about `a - b`, so there is no sound
  // decomposition and the difference itself becomes the obligation. This is
  // the rule that fires in gemm.
  EXPECT_EQ(render(prove(get("subi"), Goal::NonNegative)),
            "Obligated{subi >= 0}");
  EXPECT_EQ(render(prove(get("subi"), Goal::Positive)), "Obligated{subi > 0}");
}

TEST_F(NonNegativeProverTest, FunctionArgumentPlants) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      tt.return
    }
  )mlir");

  EXPECT_EQ(render(prove(arg(0), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
}

TEST_F(NonNegativeProverTest, LoopCarriedArgumentPlants) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c8 = arith.constant 8 : i32
      %c10 = arith.constant 10 : i32
      %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %arg0) -> (i32)
           : i32 {
        %n = arith.addi %acc, %c8 : i32 loc("in_loop")
        scf.yield %n : i32
      }
      tt.return
    }
  )mlir");

  // A block argument has no defining operation to decompose, so the recursion
  // stops there. The obligation is on the iteration argument, which no
  // loop-invariant runtime check can establish - the caller declines it.
  EXPECT_EQ(render(prove(get("in_loop"), Goal::NonNegative)),
            "Obligated{scf.for#1 >= 0}");
}

//===----------------------------------------------------------------------===//
// Structural properties: dedup, memoization, termination
//===----------------------------------------------------------------------===//

TEST_F(NonNegativeProverTest, ObligationReachedTwiceIsReportedOnce) {
  parse(R"mlir(
    tt.func @f(%arg0: i32) {
      %c2 = arith.constant 2 : i32
      %c8 = arith.constant 8 : i32
      %a = arith.addi %arg0, %c8 : i32
      %b = arith.muli %arg0, %c2 : i32
      %c = arith.addi %a, %b : i32 loc("diamond")
      tt.return
    }
  )mlir");

  // The value graph is a DAG, not a tree. Both operands of the outer addi lead
  // back to %arg0, and it must be asserted once.
  EXPECT_EQ(render(prove(get("diamond"), Goal::NonNegative)),
            "Obligated{tt.func#0 >= 0}");
}

TEST_F(NonNegativeProverTest, DeepChainTerminates) {
  // Longer than the prover's depth cap.
  std::string ir = "tt.func @f(%arg0: i32) {\n"
                   "  %c1 = arith.constant 1 : i32\n"
                   "  %v0 = arith.addi %arg0, %c1 : i32\n";
  constexpr unsigned depth = 64;
  for (unsigned i = 1; i < depth; ++i)
    ir += ("  %v" + Twine(i) + " = arith.addi %v" + Twine(i - 1) +
           ", %c1 : i32" + (i == depth - 1 ? " loc(\"top\")" : "") + "\n")
              .str();
  ir += "  tt.return\n}\n";
  parse(ir);

  // The cap must not yield a partial obligation set: "the goal holds iff these
  // hold" would be a false claim about a proof that was abandoned halfway. So
  // an over-deep chain declines outright.
  EXPECT_EQ(render(prove(get("top"), Goal::NonNegative)), "Refuted");
}

//===----------------------------------------------------------------------===//
// The gemm chain, end to end
//===----------------------------------------------------------------------===//

/// The dividend chain of `gemm_tensor_of_ptr`, reduced to the operations the
/// proof visits.
constexpr StringRef gemmChain = R"mlir(
  tt.func @gemm() {
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %group_id = arith.divui %pid, %c64 : i32
    %first_pid_m = arith.muli %group_id, %c4 : i32
    %group_size_m = arith.subi %c16, %first_pid_m : i32 loc("group_size_m")
    %gsm = arith.MINMAX %group_size_m, %c4 : i32 loc("gsm")
    %pid_m = arith.remui %pid, %c64 : i32
    %pid_n = arith.divsi %pid_m, %gsm : i32
    %offs_bn = arith.muli %pid_n, %c256 : i32
    %splat = tt.splat %offs_bn : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offs_bn_11 = arith.addi %splat, %range
                  : tensor<256xi32> loc("offs_bn_11")
    tt.return
  }
)mlir";

TEST_F(NonNegativeProverTest, GemmChainPlantsOneScalarObligation) {
  std::string ir = gemmChain.str();
  ir.replace(ir.find("MINMAX"), 6, "minsi");
  parse(ir);

  // The whole design in one assertion. Three things are pinned:
  //
  //  - the obligation is a single scalar, defined at the top level, whose
  //    backward closure is grid arithmetic over constants;
  //  - it lands on %group_size_m and not on %gsm, because the recursion passes
  //    *through* the minsi (min(a,b) > 0 iff a > 0 and b > 0 is sound) and
  //    stops at the subi, where no sound rule exists. Both facts unblock the
  //    analysis, but the deeper one is reached without a walk-up heuristic and
  //    is the better diagnostic;
  //  - the goal is Positive, so the emitted check is `sgt 0`. With `sge` the
  //    divisor range becomes [0, 4], which contains zero, and inferDivS bails.
  EXPECT_EQ(render(prove(get("offs_bn_11"), Goal::NonNegative)),
            "Obligated{group_size_m > 0}");

  // The intermediate divisor is not itself the obligation, but it is the value
  // whose unknown sign blocks the analysis.
  EXPECT_EQ(render(prove(get("gsm"), Goal::Positive)),
            "Obligated{group_size_m > 0}");
}

TEST_F(NonNegativeProverTest, GemmChainWithMaxSINeedsNoObligation) {
  std::string ir = gemmChain.str();
  ir.replace(ir.find("MINMAX"), 6, "maxsi");
  parse(ir);

  // With maxsi the divisor is at least 4 unconditionally, so the range
  // analysis answers on its own and the whole chain is proven with nothing to
  // assert. The pass converts these for free; only the minsi form pays for a
  // runtime check.
  EXPECT_EQ(render(prove(get("offs_bn_11"), Goal::NonNegative)), "Satisfied");
}

} // namespace
