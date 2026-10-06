//===- SignedDivRemDeductionTest.cpp --------------------------------------===//
//
// Unit tests for triton::intel::signedDivRemDeductionApplies, which mirrors the
// two AxisInfo deduction sites that speculation can recover:
//
//   1. DivOpAxisInfoVisitor::getConstancy  - contiguous lhs, constant rhs
//   2. RemOpAxisInfoVisitor::getContiguity - contiguous lhs, constant rhs
//
// These tests exist so the predicate cannot drift away from the conditions it
// mirrors. Each covers both sides of every condition.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/SignedDivRemDeduction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using mlir::triton::intel::signedDivRemDeductionApplies;

namespace {

class SignedDivRemDeductionTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    module = ModuleOp::create(UnknownLoc::get(&ctx));
  }

  /// Builds `OpTy` over two fresh operands of type `ty`. The operation is kept
  /// alive by `module` for the lifetime of the test.
  template <typename OpTy> OpTy makeOp(Type ty) {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToEnd(module->getBody());
    Location loc = builder.getUnknownLoc();
    auto funcOp = func::FuncOp::create(loc, "f" + std::to_string(numFuncs++),
                                       builder.getFunctionType({ty, ty}, {ty}));
    module->push_back(funcOp);
    Block *entry = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto op = OpTy::create(builder, loc, entry->getArgument(0),
                           entry->getArgument(1));
    func::ReturnOp::create(builder, loc, ValueRange{op.getResult()});
    return op;
  }

  Type tensorTy(ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, IntegerType::get(&ctx, 32));
  }

  Type scalarTy() { return IntegerType::get(&ctx, 32); }

protected:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  unsigned numFuncs = 0;
};

//===----------------------------------------------------------------------===//
// Contiguous dividend, constant divisor, gcd > 1.
//===----------------------------------------------------------------------===//

TEST_F(SignedDivRemDeductionTest, DivContiguousLhsConstantRhs) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({128}));
  AxisInfo lhs(/*contiguity=*/{128}, /*divisibility=*/{128}, /*constancy=*/{1});
  AxisInfo rhs(/*contiguity=*/{1}, /*divisibility=*/{4}, /*constancy=*/{128});
  EXPECT_TRUE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, RemContiguousLhsConstantRhs) {
  auto op = makeOp<arith::RemSIOp>(tensorTy({128}));
  AxisInfo lhs({128}, {128}, {1});
  AxisInfo rhs({1}, {4}, {128});
  EXPECT_TRUE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, DivNonContiguousLhs) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({128}));
  AxisInfo lhs(/*contiguity=*/{1}, {128}, {1});
  AxisInfo rhs({1}, {4}, {128});
  EXPECT_FALSE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, DivNonConstantRhs) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({128}));
  AxisInfo lhs({128}, {128}, {1});
  AxisInfo rhs({1}, {4}, /*constancy=*/{1});
  EXPECT_FALSE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, DivGcdIsOne) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({128}));
  AxisInfo lhs({128}, /*divisibility=*/{1}, {1});
  AxisInfo rhs({1}, /*divisibility=*/{1}, {128});
  EXPECT_FALSE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, DivMatchesOnInnerDimensionOnly) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({1, 64}));
  AxisInfo lhs(/*contiguity=*/{1, 64}, /*divisibility=*/{1, 64},
               /*constancy=*/{1, 1});
  AxisInfo rhs(/*contiguity=*/{1, 1}, /*divisibility=*/{1, 128},
               /*constancy=*/{1, 64});
  // Dimension 0 matches the shape but has contiguity 1, so its gcd is 1;
  // dimension 1 is the one that carries the deduction.
  EXPECT_TRUE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, DivUnitDimensionAloneIsNotEnough) {
  auto op = makeOp<arith::DivSIOp>(tensorTy({1, 64}));
  AxisInfo lhs(/*contiguity=*/{1, 1}, {1, 64}, {1, 1});
  AxisInfo rhs(/*contiguity=*/{1, 1}, {1, 128}, /*constancy=*/{1, 64});
  EXPECT_FALSE(signedDivRemDeductionApplies(op, lhs, rhs));
}

TEST_F(SignedDivRemDeductionTest, ScalarsNeverMatch) {
  // Both sites index shape[dim], so they bail when the result is not a
  // RankedTensorType.
  AxisInfo lhs({1}, {64}, {1});
  AxisInfo rhs({1}, {8}, /*constancy=*/{2});
  EXPECT_FALSE(signedDivRemDeductionApplies(makeOp<arith::RemSIOp>(scalarTy()),
                                            lhs, rhs));
  EXPECT_FALSE(signedDivRemDeductionApplies(makeOp<arith::DivSIOp>(scalarTy()),
                                            lhs, rhs));
}

//===----------------------------------------------------------------------===//
// Only the signed operations are candidates.
//===----------------------------------------------------------------------===//

TEST_F(SignedDivRemDeductionTest, UnsignedOpsAreNotCandidates) {
  AxisInfo lhs({128}, {128}, {1});
  AxisInfo rhs({1}, {4}, {128});
  EXPECT_FALSE(signedDivRemDeductionApplies(
      makeOp<arith::DivUIOp>(tensorTy({128})), lhs, rhs));
  EXPECT_FALSE(signedDivRemDeductionApplies(
      makeOp<arith::RemUIOp>(tensorTy({128})), lhs, rhs));
  EXPECT_FALSE(signedDivRemDeductionApplies(
      makeOp<arith::AddIOp>(tensorTy({128})), lhs, rhs));
}

} // namespace
