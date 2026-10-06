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
  EXPECT_EQ(verdict(get("cmp")), "Unknown"); // never Refuted (Review Focus 1)
}

// E1 of the design: the inductor reduction shape the census shows exiting
// walk 1 at iv-range-unknown.
static const char *kE1 = R"(
  tt.func @e1(%ptr: !tt.ptr<f32>, %rnumel: i32) {
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

TEST_F(SymbolicBoundsTest, E1_ConditionalOnExactLoopEnd) {
  parse(kE1);
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
  std::string ir = kE1;
  ir.replace(ir.find("scf.for %r"), 10, "scf.for unsigned %r");
  parse(ir);
  // Loop contract (ii): NonNegative(lb) discharges because lb = 0 is a
  // constant; NonNegative(ub) and AtMost(ub, INT_MAX - step + 1) remain.
  // Order: fact, then preconditions.
  EXPECT_EQ(verdict(get("mask")),
            "Conditional{arg1 divisible by 64; arg1 >= 0; arg1 <= 2147483584}");
}

TEST_F(SymbolicBoundsTest, RefutedOnlyOnConstantHi) {
  // pid >= 0 from the range analysis, so hi(d) = -pid <= 0 < 1 by sign
  // reasoning; 4.3 step 3 refutes only on a constant hi(d).
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

} // namespace
