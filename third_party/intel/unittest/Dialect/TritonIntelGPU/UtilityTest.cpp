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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton::gpu::intel;
namespace ttg = mlir::triton::gpu;

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

// getScatterBlockWriteRuns (#8868): one fixture per rule of its contract.
class ScatterBlockWriteRunsTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<ttg::TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
  }

  RankedTensorType dpas(ArrayRef<int64_t> shape, unsigned threadsPerWarp) {
    auto enc = DpasEncodingAttr::get(
        &ctx, /*repeatCount=*/8, /*systolicDepth=*/8, /*executionSize=*/16,
        /*opsPerChannel=*/2, /*warpsPerCTA=*/{2, 2}, /*repCluster=*/{2, 2},
        threadsPerWarp);
    return RankedTensorType::get(shape, BFloat16Type::get(&ctx), enc);
  }

  RankedTensorType blocked(ArrayRef<int64_t> shape, Type elemTy,
                           ArrayRef<unsigned> sizePerThread,
                           ArrayRef<unsigned> threadsPerWarp,
                           SmallVector<unsigned, 2> warpsPerCTA = {1, 1}) {
    auto enc = ttg::BlockedEncodingAttr::get(
        &ctx, sizePerThread, threadsPerWarp, warpsPerCTA, /*order=*/{1, 0},
        ttg::CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2));
    return RankedTensorType::get(shape, elemTy, enc);
  }

  // A bf16 layout whose 16 lanes hold consecutive columns.
  RankedTensorType generic(ArrayRef<int64_t> shape,
                           std::vector<std::vector<int32_t>> regBases,
                           std::vector<std::vector<int32_t>> warpBases) {
    auto S = [&](StringRef name) { return StringAttr::get(&ctx, name); };
    triton::LinearLayout ll({{S("register"), regBases},
                             {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                             {S("warp"), warpBases},
                             {S("block"), {}}},
                            {S("dim0"), S("dim1")});
    return RankedTensorType::get(
        shape, BFloat16Type::get(&ctx),
        ttg::GenericLinearEncodingAttr::get(&ctx, std::move(ll)));
  }

  // (row, col, regs) of each run, sorted: the contract leaves their order open.
  using Run = std::tuple<int32_t, int32_t, SmallVector<unsigned>>;
  static std::vector<Run> sorted(ArrayRef<ScatterBlockWriteRun> runs) {
    std::vector<Run> result;
    for (const ScatterBlockWriteRun &run : runs)
      result.emplace_back(run.row, run.col, run.regs);
    llvm::sort(result);
    return result;
  }

protected:
  MLIRContext ctx;
};

TEST_F(ScatterBlockWriteRunsTest, AcceptsDpasBf16) {
  auto runs = getScatterBlockWriteRuns(dpas({32, 64}, 16));
  ASSERT_TRUE(runs.has_value());
  // Each row of warp 0 holds two registers, at columns 0 and 16.
  EXPECT_EQ(sorted(*runs), (std::vector<Run>{{0, 0, {0, 8}},
                                             {1, 0, {1, 9}},
                                             {2, 0, {2, 10}},
                                             {3, 0, {3, 11}},
                                             {4, 0, {4, 12}},
                                             {5, 0, {5, 13}},
                                             {6, 0, {6, 14}},
                                             {7, 0, {7, 15}},
                                             {8, 0, {16, 24}},
                                             {9, 0, {17, 25}},
                                             {10, 0, {18, 26}},
                                             {11, 0, {19, 27}},
                                             {12, 0, {20, 28}},
                                             {13, 0, {21, 29}},
                                             {14, 0, {22, 30}},
                                             {15, 0, {23, 31}}}));
}

TEST_F(ScatterBlockWriteRunsTest, AcceptsBlockedUnitColumns) {
  auto runs = getScatterBlockWriteRuns(
      blocked({16, 64}, Float32Type::get(&ctx), {1, 1}, {1, 16}));
  ASSERT_TRUE(runs.has_value());
  EXPECT_EQ(runs->size(), 16u);
  for (const ScatterBlockWriteRun &run : *runs)
    EXPECT_EQ(run.regs.size(), 4u);
}

TEST_F(ScatterBlockWriteRunsTest, SplitsLongRuns) {
  // One row of 16 registers, 16 columns apart, is cut into 8 + 8.
  auto runs = getScatterBlockWriteRuns(
      blocked({1, 256}, BFloat16Type::get(&ctx), {1, 1}, {1, 16}));
  ASSERT_TRUE(runs.has_value());
  EXPECT_EQ(sorted(*runs),
            (std::vector<Run>{{0, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
                              {0, 128, {8, 9, 10, 11, 12, 13, 14, 15}}}));
}

TEST_F(ScatterBlockWriteRunsTest, SortsAndSplitsColumns) {
  // Registers 0-3 sit at columns 0, 64, 16 and 80.
  auto runs = getScatterBlockWriteRuns(
      generic({1, 128}, {{0, 64}, {0, 16}}, {{0, 32}}));
  ASSERT_TRUE(runs.has_value());
  EXPECT_EQ(sorted(*runs), (std::vector<Run>{{0, 0, {0, 2}}, {0, 64, {1, 3}}}));
}

TEST_F(ScatterBlockWriteRunsTest, RejectsLaneRows) {
  // Lane 16 starts row 1.
  EXPECT_FALSE(getScatterBlockWriteRuns(dpas({32, 64}, 32)).has_value());
}

TEST_F(ScatterBlockWriteRunsTest, RejectsStridedLanes) {
  // Lane l holds columns 2l and 2l + 1.
  EXPECT_FALSE(getScatterBlockWriteRuns(
                   blocked({16, 64}, BFloat16Type::get(&ctx), {1, 2}, {1, 16}))
                   .has_value());
}

TEST_F(ScatterBlockWriteRunsTest, RejectsWidth) {
  for (unsigned width : {8u, 64u})
    EXPECT_FALSE(
        getScatterBlockWriteRuns(
            blocked({16, 64}, IntegerType::get(&ctx, width), {1, 1}, {1, 16}))
            .has_value())
        << "i" << width;
}

TEST_F(ScatterBlockWriteRunsTest, AcceptsSixteenByteSubgroup) {
  EXPECT_TRUE(getScatterBlockWriteRuns(
                  blocked({16, 64}, BFloat16Type::get(&ctx), {1, 1}, {1, 8}))
                  .has_value());
}

TEST_F(ScatterBlockWriteRunsTest, RejectsNarrowSubgroup) {
  // 4 lanes x 2 bytes: warp 1 starts 8 bytes in.
  EXPECT_FALSE(
      getScatterBlockWriteRuns(
          blocked({16, 64}, BFloat16Type::get(&ctx), {1, 1}, {1, 4}, {1, 2}))
          .has_value());
}

TEST_F(ScatterBlockWriteRunsTest, RejectsSwizzledWarp) {
  // Warp 1 holds register 1 left of register 0.
  EXPECT_FALSE(getScatterBlockWriteRuns(generic({2, 32}, {{0, 16}}, {{1, 16}}))
                   .has_value());
}

TEST_F(ScatterBlockWriteRunsTest, RejectsOverlappingBases) {
  // Register 1 and lane 1 hold the same element.
  EXPECT_FALSE(
      getScatterBlockWriteRuns(generic({1, 16}, {{0, 1}}, {})).has_value());
}

TEST_F(ScatterBlockWriteRunsTest, SkipsReplicatedRegisters) {
  // At 16 columns, registers 8-15 and 24-31 replicate 0-7 and 16-23.
  auto runs = getScatterBlockWriteRuns(dpas({16, 16}, 16));
  ASSERT_TRUE(runs.has_value());
  SmallVector<unsigned> regs;
  for (const ScatterBlockWriteRun &run : *runs)
    llvm::append_range(regs, run.regs);
  llvm::sort(regs);
  EXPECT_EQ(regs, (SmallVector<unsigned>{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                                         20, 21, 22, 23}));
}

} // namespace
