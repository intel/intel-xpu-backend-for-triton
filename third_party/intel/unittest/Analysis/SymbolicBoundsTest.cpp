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

} // namespace
