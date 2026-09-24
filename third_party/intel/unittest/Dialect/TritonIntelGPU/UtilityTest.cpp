// Tests for ttgi::isDivisible (issues/8073).
//
// The divisor used to be `unsigned` and was divided by unguarded, so a zero
// divisor raised SIGFPE, and an i64 divisor wider than 32 bits was narrowed
// either to zero (SIGFPE) or to one (a false divisibility proof). Constants
// were also zero-extended, so a negative constant was tested as a huge
// unsigned value.

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

} // namespace
