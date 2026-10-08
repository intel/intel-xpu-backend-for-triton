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
//
// An scf.for iteration argument advanced by a loop-invariant amount,
// `x = init; x += c`, is divisible when both `init` and `c` are, in a nested
// loop too (#8060).

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton::gpu::intel;
namespace tt = mlir::triton;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

class IsDivisibleTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
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

  // The iteration argument `%x` of
  //   scf.for %iv = 0 to 128 step 32 iter_args(%x = init) {
  //     scf.yield %x + inc
  //   }
  BlockArgument iterArgAdvancedBy(Value init, Value inc) {
    auto forOp = scf::ForOp::create(builder, UnknownLoc::get(&ctx),
                                    constant(0, 32), constant(128, 32),
                                    constant(32, 32), ValueRange{init});
    BlockArgument x = forOp.getRegionIterArgs()[0];
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    scf::YieldOp::create(builder, UnknownLoc::get(&ctx),
                         binary<arith::AddIOp>(x, inc));
    return x;
  }

protected:
  MLIRContext ctx;
  OpBuilder builder{&ctx};
  OwningOpRef<ModuleOp> module;
};

class IntelLoadShuffleBitcastOpTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<tt::TritonDialect>();
    ctx.getOrLoadDialect<ttgi::TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
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

TEST_F(IsDivisibleTest, LoopCarriedAddRequiresInitAndIncrement) {
  EXPECT_TRUE(
      isDivisible(iterArgAdvancedBy(constant(32, 32), constant(16, 32)), 16));
  // The first iteration sees `init` itself, whatever the increment (#7990).
  EXPECT_FALSE(
      isDivisible(iterArgAdvancedBy(constant(1, 32), constant(16, 32)), 16));
  EXPECT_FALSE(
      isDivisible(iterArgAdvancedBy(constant(32, 32), constant(1, 32)), 16));
}

TEST_F(IsDivisibleTest, LoopCarriedAddInNestedLoop) {
  // A K loop in a persistent or stream-K tile loop. The nested-loop refusal
  // applies to the inner induction variable, not to its iteration arguments.
  auto outer =
      scf::ForOp::create(builder, UnknownLoc::get(&ctx), constant(0, 32),
                         constant(128, 32), constant(64, 32));
  builder.setInsertionPointToStart(outer.getBody());
  BlockArgument x = iterArgAdvancedBy(constant(0, 32), constant(32, 32));
  EXPECT_TRUE(isDivisible(x, 16));
  auto inner = cast<scf::ForOp>(x.getOwner()->getParentOp());
  EXPECT_FALSE(isDivisible(inner.getInductionVar(), 16));
}

TEST_F(IntelLoadShuffleBitcastOpTest, ReportsReadAndVolatileWriteEffects) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  tt.func @test(%arg0: !tt.ptr<f16>) {
    %0 = ttig.load_shuffle_bitcast %arg0 : !tt.ptr<f16>
    tt.return
  }
}
)mlir",
                                            &ctx);
  ASSERT_TRUE(module) << "failed to parse test module";

  Operation *loadOp = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "ttig.load_shuffle_bitcast")
      loadOp = op;
  });
  ASSERT_NE(loadOp, nullptr);

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(loadOp);
  ASSERT_TRUE(effectOp);

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  EXPECT_TRUE(llvm::any_of(effects, [](const auto &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect());
  }));
  EXPECT_FALSE(llvm::any_of(effects, [](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  }));

  loadOp->setAttr("isVolatile", builder->getBoolAttr(true));
  effects.clear();
  effectOp.getEffects(effects);
  EXPECT_TRUE(llvm::any_of(effects, [](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  }));
}

} // namespace
