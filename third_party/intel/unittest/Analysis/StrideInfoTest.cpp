#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::intel;

namespace {

class StrideInfoTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    ctx.getOrLoadDialect<TritonGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  ModuleOp buildModule() {
    Location loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    module->setAttr(AttrNumWarpsName, builder->getI32IntegerAttr(4));
    module->setAttr(AttrNumThreadsPerWarp, builder->getI32IntegerAttr(16));
    return module;
  }

  func::FuncOp buildFunc(ModuleOp module, StringRef name = "test_func") {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToEnd(module.getBody());
    FunctionType funcType = builder->getFunctionType({}, {});
    auto funcOp =
        func::FuncOp::create(builder->getUnknownLoc(), name, funcType);
    module.push_back(funcOp);
    Block *entry = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(entry);
    func::ReturnOp::create(*builder, builder->getUnknownLoc());
    return funcOp;
  }

  scf::ForOp buildScfFor(OpBuilder &b, int64_t lowerBound, int64_t upperBound,
                         int64_t step) {
    Location loc = b.getUnknownLoc();
    auto lb = arith::ConstantIndexOp::create(b, loc, lowerBound);
    auto ub = arith::ConstantIndexOp::create(b, loc, upperBound);
    auto stepOp = arith::ConstantIndexOp::create(b, loc, step);
    auto forOp = scf::ForOp::create(b, loc, lb, ub, stepOp);
    b.setInsertionPointToStart(forOp.getBody());
    return forOp;
  }

  BlockedEncodingAttr makeBlocked(ArrayRef<unsigned> sizePerThread,
                                  ArrayRef<unsigned> threadsPerWarp,
                                  ArrayRef<unsigned> warpsPerCTA,
                                  ArrayRef<unsigned> order) {
    auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, order.size());
    return BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                    warpsPerCTA, order, cgaLayout);
  }

  BlockArgument addPtrArg(func::FuncOp funcOp, Type elemTy,
                          unsigned addressSpace = 1) {
    Block *block = &funcOp.getBody().front();
    auto ptrTy = PointerType::get(elemTy, addressSpace);
    FunctionType newFuncType = builder->getFunctionType(
        {ptrTy}, funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
    return block->addArgument(ptrTy, builder->getUnknownLoc());
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// Test 1: Constant scalar has IV stride 0 for any loop
TEST_F(StrideInfoTest, ConstantScalarIVStrideZero) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  // Build a constant
  Location loc = builder->getUnknownLoc();
  auto constOp =
      arith::ConstantIntOp::create(*builder, loc, builder->getI64Type(), 42);

  // Build a dummy loop
  auto forOp = buildScfFor(*builder, 0, 10, 1);
  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(constOp);
  ASSERT_NE(info, nullptr);

  // IV stride should be 0 or nullptr (both acceptable per plan)
  const auto *ivStrideVec = info->getIVStride(forOp);
  if (ivStrideVec != nullptr) {
    EXPECT_EQ((*ivStrideVec)[0], 0);
  }
  // Also check getIVStride(loop, dim) returns 0
  EXPECT_EQ(info->getIVStride(forOp, 0), 0);
}

// Test 2: Loop IV has stride 1 w.r.t. its own loop, 0 w.r.t. other loop
TEST_F(StrideInfoTest, LoopIVOwnLoop) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  // Build first loop
  auto forOp1 = buildScfFor(*builder, 0, 10, 1);
  Value iv1 = forOp1.getInductionVar();
  scf::YieldOp::create(*builder, loc);

  // Build second independent loop
  builder->setInsertionPointAfter(forOp1);
  auto forOp2 = buildScfFor(*builder, 0, 10, 1);
  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(iv1);
  ASSERT_NE(info, nullptr);

  // IV stride w.r.t. its own loop should be 1
  EXPECT_EQ(info->getIVStride(forOp1, 0), 1);

  // IV stride w.r.t. unrelated loop should be 0
  EXPECT_EQ(info->getIVStride(forOp2, 0), 0);
}

// Test 3: arith.muli(iv, 128) → IV stride 128
TEST_F(StrideInfoTest, MulIVByConstant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  auto forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // iv * 128 — iv is index-typed, so use ConstantIndexOp for the multiplier.
  auto c128 = arith::ConstantIndexOp::create(*builder, loc, 128);
  auto mulOp = arith::MulIOp::create(*builder, loc, iv, c128);

  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(mulOp);
  ASSERT_NE(info, nullptr);

  EXPECT_EQ(info->getIVStride(forOp, 0), 128);
}

// Test 4: arith.addi(iv, const) → IV stride 1
TEST_F(StrideInfoTest, AddIVConstant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  auto forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // iv + 5 — iv is index-typed.
  auto c5 = arith::ConstantIndexOp::create(*builder, loc, 5);
  auto addOp = arith::AddIOp::create(*builder, loc, iv, c5);

  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(addOp);
  ASSERT_NE(info, nullptr);

  EXPECT_EQ(info->getIVStride(forOp, 0), 1);
}

// Test 5: tt.splat(muli(iv, 128)) → IV stride [128]
TEST_F(StrideInfoTest, SplatMulIV) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  auto forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // iv * 128 — iv is index-typed; cast the product to i64 before splatting.
  auto c128 = arith::ConstantIndexOp::create(*builder, loc, 128);
  auto mulOp = arith::MulIOp::create(*builder, loc, iv, c128);
  Type i64Ty = builder->getI64Type();
  auto mulI64 = arith::IndexCastOp::create(*builder, loc, i64Ty, mulOp);

  // splat to tensor<128xi64>
  auto encoding = makeBlocked({1}, {16}, {4}, {0});
  auto tensorTy = RankedTensorType::get({128}, i64Ty, encoding);
  auto splatOp = SplatOp::create(*builder, loc, tensorTy, mulI64);

  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(splatOp);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getRank(), 1u);

  // IV stride should be [128] for dim 0
  EXPECT_EQ(info->getIVStride(forOp, 0), 128);
}

// Test 6: 1-D streaming case: addptr(splat(base), splat(muli(iv, 128)))
TEST_F(StrideInfoTest, OneDimStreamingAddPtr) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);

  // Add a pointer argument
  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  builder->setInsertionPointToStart(&funcOp.getBody().front());
  Location loc = builder->getUnknownLoc();

  auto forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // iv * 128 — iv is index-typed; cast the product to i64 for the offset splat.
  auto c128 = arith::ConstantIndexOp::create(*builder, loc, 128);
  auto mulOp = arith::MulIOp::create(*builder, loc, iv, c128);
  Type i64Ty = builder->getI64Type();
  auto mulI64 = arith::IndexCastOp::create(*builder, loc, i64Ty, mulOp);

  // Create tensor<128x!tt.ptr<f16>> type
  auto encoding = makeBlocked({1}, {16}, {4}, {0});
  auto ptrTy = PointerType::get(f16Ty, 1);
  auto tensorPtrTy = RankedTensorType::get({128}, ptrTy, encoding);

  // splat(base_ptr)
  auto baseSplat = SplatOp::create(*builder, loc, tensorPtrTy, basePtr);

  // splat(muli(iv, 128))
  auto tensorI64Ty = RankedTensorType::get({128}, i64Ty, encoding);
  auto offsetSplat = SplatOp::create(*builder, loc, tensorI64Ty, mulI64);

  // addptr
  auto addPtrOp =
      AddPtrOp::create(*builder, loc, tensorPtrTy, baseSplat, offsetSplat);

  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(addPtrOp);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getRank(), 1u);

  // IV stride should be 128 (streaming access)
  EXPECT_EQ(info->getIVStride(forOp, 0), 128);

  // Spatial stride should be 0 (base is splatted)
  EXPECT_EQ(info->getStride(0), 0);
}

// Test 7: Nested loops — inner IV queried for outer-only-dependent value
TEST_F(StrideInfoTest, NestedLoopsInnerIVOuterValue) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  // Outer loop
  auto outerForOp = buildScfFor(*builder, 0, 10, 1);
  Value outerIV = outerForOp.getInductionVar();

  // outerIV * 16 — outerIV is index-typed.
  auto c16 = arith::ConstantIndexOp::create(*builder, loc, 16);
  auto mulOp = arith::MulIOp::create(*builder, loc, outerIV, c16);

  // Inner loop
  auto innerForOp = buildScfFor(*builder, 0, 5, 1);
  scf::YieldOp::create(*builder, loc);

  // Close outer loop
  builder->setInsertionPointAfter(innerForOp);
  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(mulOp);
  ASSERT_NE(info, nullptr);

  // Query w.r.t. inner loop: should be 0 (outer-IV-derived, doesn't change in
  // inner)
  EXPECT_EQ(info->getIVStride(innerForOp, 0), 0);

  // Query w.r.t. outer loop: should be 16
  EXPECT_EQ(info->getIVStride(outerForOp, 0), 16);
}

// scf.while IV-stride seeding is intentionally deferred: its before-region
// args are control-flow-fed from the init operands, which bypasses the
// `visitNonControlFlowArguments` hook used to seed the scf.for IV column.
// Handling it requires overriding the solver's region-successor propagation,
// which is out of scope for the streaming-pattern regression this PR targets.
// PR-B (TemporalReuseAnalysis) will revisit scf.while when a consumer
// actually exercises it.

// Test 8: Spatial stride regression — tt.make_range
TEST_F(StrideInfoTest, SpatialRangeStrideOne) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  // tt.make_range {start=0, end=128} : tensor<128xi32>
  auto encoding = makeBlocked({1}, {16}, {4}, {0});
  auto tensorTy = RankedTensorType::get({128}, builder->getI32Type(), encoding);
  auto rangeOp = MakeRangeOp::create(*builder, loc, tensorTy, 0, 128);

  // Build a dummy loop to test IV stride is empty/zero
  auto forOp = buildScfFor(*builder, 0, 10, 1);
  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(rangeOp);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getRank(), 1u);

  // Spatial stride should be 1
  EXPECT_EQ(info->getStride(0), 1);

  // IV stride should be 0 or nullptr (not loop-dependent)
  const auto *ivStrideVec = info->getIVStride(forOp);
  if (ivStrideVec != nullptr) {
    EXPECT_EQ((*ivStrideVec)[0], 0);
  }
}

// Test 9: Spatial stride regression — tt.splat(const)
TEST_F(StrideInfoTest, SpatialSplatConstant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Location loc = builder->getUnknownLoc();

  // Constant
  auto c0 =
      arith::ConstantIntOp::create(*builder, loc, builder->getI32Type(), 0);

  // tt.splat(c0) : tensor<128xi32>
  auto encoding = makeBlocked({1}, {16}, {4}, {0});
  auto tensorTy = RankedTensorType::get({128}, builder->getI32Type(), encoding);
  auto splatOp = SplatOp::create(*builder, loc, tensorTy, c0);

  // Build a dummy loop
  auto forOp = buildScfFor(*builder, 0, 10, 1);
  scf::YieldOp::create(*builder, loc);

  // Run analysis
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  ModuleStrideAnalysis strideAnalysis(module, axisInfo);

  StrideInfo *info = strideAnalysis.getStrideInfo(splatOp);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getRank(), 1u);

  // Spatial stride should be 0 (splat)
  EXPECT_EQ(info->getStride(0), 0);

  // IV stride should be 0 (constant value)
  EXPECT_EQ(info->getIVStride(forOp, 0), 0);
}

} // namespace
