// Tests for ttgi::isDivisible (issues/8073).
//
// The divisor used to be `unsigned` and was divided by unguarded, so a zero
// divisor raised SIGFPE, and an i64 divisor wider than 32 bits was narrowed
// either to zero (SIGFPE) or to one (a false divisibility proof). Constants
// were also zero-extended, so a negative constant was tested as a huge
// unsigned value.
//
// isDivisible also looks through arith.subi, arith.minsi, arith.maxsi and
// arith.select, requiring every value that can reach the result to be
// divisible.

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton::gpu::intel;

namespace {

class IsDivisibleTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module->getBody());
  }

  Value constant(int64_t value, unsigned width) {
    return arith::ConstantIntOp::create(builder, UnknownLoc::get(&ctx), value,
                                        width);
  }

  template <typename OpTy> Value binary(Value lhs, Value rhs) {
    return OpTy::create(builder, UnknownLoc::get(&ctx), lhs, rhs);
  }

  Value select(Value trueValue, Value falseValue) {
    Value cond = arith::ConstantIntOp::create(builder, UnknownLoc::get(&ctx), 1,
                                              /*width=*/1);
    return arith::SelectOp::create(builder, UnknownLoc::get(&ctx), cond,
                                   trueValue, falseValue);
  }

protected:
  MLIRContext ctx;
  OpBuilder builder{&ctx};
  OwningOpRef<ModuleOp> module;
};

TEST_F(IsDivisibleTest, NonPositiveDivisorIsRejected) {
  // A non-positive divisor is a caller bug: it asserts in debug builds and is
  // answered conservatively in release builds.
  Value c16 = constant(16, 64);
  EXPECT_DEBUG_DEATH(EXPECT_FALSE(isDivisible(c16, 0)),
                     "Expecting a positive divisor");
  EXPECT_DEBUG_DEATH(EXPECT_FALSE(isDivisible(c16, -16)),
                     "Expecting a positive divisor");
}

TEST_F(IsDivisibleTest, DivisorWiderThan32Bits) {
  // 2^32 narrowed to `unsigned` is 0.
  EXPECT_TRUE(isDivisible(constant(int64_t(1) << 33, 64), int64_t(1) << 32));
  // 2^32 + 1 narrowed to `unsigned` is 1, which proved anything divisible.
  EXPECT_FALSE(isDivisible(constant(3, 64), (int64_t(1) << 32) + 1));
}

TEST_F(IsDivisibleTest, NegativeConstantIsSigned) {
  // Zero-extended, i32 -12 is 4294967284, which is not a multiple of 3.
  EXPECT_TRUE(isDivisible(constant(-12, 32), 3));
  EXPECT_FALSE(isDivisible(constant(-13, 32), 3));
}

TEST_F(IsDivisibleTest, SubIRequiresBothOperands) {
  EXPECT_TRUE(isDivisible(
      binary<arith::SubIOp>(constant(48, 32), constant(16, 32)), 16));
  EXPECT_FALSE(isDivisible(
      binary<arith::SubIOp>(constant(48, 32), constant(1, 32)), 16));
  EXPECT_FALSE(isDivisible(
      binary<arith::SubIOp>(constant(1, 32), constant(48, 32)), 16));
}

TEST_F(IsDivisibleTest, MinMaxRequireBothOperands) {
  EXPECT_TRUE(isDivisible(
      binary<arith::MinSIOp>(constant(48, 32), constant(16, 32)), 16));
  EXPECT_FALSE(isDivisible(
      binary<arith::MinSIOp>(constant(48, 32), constant(1, 32)), 16));
  EXPECT_TRUE(isDivisible(
      binary<arith::MaxSIOp>(constant(48, 32), constant(16, 32)), 16));
  EXPECT_FALSE(isDivisible(
      binary<arith::MaxSIOp>(constant(1, 32), constant(48, 32)), 16));
}

TEST_F(IsDivisibleTest, SelectRequiresBothValues) {
  EXPECT_TRUE(isDivisible(select(constant(48, 32), constant(16, 32)), 16));
  // The condition is not consulted, even though it is the constant `true`.
  EXPECT_FALSE(isDivisible(select(constant(48, 32), constant(1, 32)), 16));
  EXPECT_FALSE(isDivisible(select(constant(1, 32), constant(48, 32)), 16));
}

} // namespace
