#include "intel/include/Analysis/SpatialReuseAnalysis.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace {

class SpatialReuseAnalysisTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  DpasEncodingAttr
  makeDpas(ArrayRef<unsigned> warps = {2, 2}, unsigned repeatCount = 8,
           unsigned systolicDepth = 8, unsigned executionSize = 16,
           unsigned opsPerChannel = 2, ArrayRef<unsigned> repCluster = {1, 1},
           unsigned threadsPerWarp = 16) {
    return DpasEncodingAttr::get(&ctx, repeatCount, systolicDepth,
                                 executionSize, opsPerChannel, warps,
                                 repCluster, threadsPerWarp, std::nullopt);
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
    funcOp.addEntryBlock();
    return funcOp;
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

TEST_F(SpatialReuseAnalysisTest, BlockedWarpsPartition) {
  // Blocked encoding with warpsPerCTA=[4,1]. warpsPerCTA[1] == 1 so dim 1
  // is warp-invariant (all warps see the same N coord); dim 0 is partitioned.
  // Expected axis set: {1}. hasCrossSubgroupReuse: true.
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 32};
  SmallVector<unsigned> warpsPerCTA = {4, 1};
  SmallVector<unsigned> order = {1, 0};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked = BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                          warpsPerCTA, order, cgaLayout);
  auto ty = RankedTensorType::get({32, 32}, builder->getF32Type(), blocked);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, DotOperandDpasOpA) {
  // DotOperand{Dpas, opIdx=0}: K (dim 1) is warp-broadcast, M (dim 0)
  // is warp-partitioned. See DPAStoLinearLayout:396-397.
  // Expected axis set: {1}. hasCrossSubgroupReuse: true.
  DpasEncodingAttr dpas = makeDpas();
  auto dotOpEnc =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  auto ty = RankedTensorType::get({64, 64}, builder->getF32Type(), dotOpEnc);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, DotOperandDpasOpB) {
  // DotOperand{Dpas, opIdx=1}: K (dim 0) is warp-broadcast, N (dim 1)
  // is warp-partitioned. See DPAStoLinearLayout:417-420.
  // Expected axis set: {0}. hasCrossSubgroupReuse: true.
  DpasEncodingAttr dpas = makeDpas();
  auto dotOpEnc =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, dpas, /*kWidth=*/2);
  auto ty = RankedTensorType::get({64, 64}, builder->getF32Type(), dotOpEnc);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(0u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, BlockScaledScaleOperand) {
  // Requires PR #6825 (BlockScaledDPAS warp-broadcast fix) to land first.
  GTEST_SKIP() << "Depends on PR #6825: BlockScaledDPAS scale operand "
                  "warp basis fix";
  // LinearEncoding from BlockScaledDPAStoLinearLayout for MXFP8 scale operand.
  // All output dims must be warp-invariant after the preparatory
  // DPAStoLinearLayoutTest::BlockScaledScaleOperandWarpBroadcast audit.
  // Expected axis set: full {0, 1}. hasCrossSubgroupReuse: true.
  DpasEncodingAttr dpas = makeDpas();
  ArrayRef<int64_t> shape = {128, 2};
  LinearLayout ll = BlockScaledDPAStoLinearLayout(shape, dpas, /*opIdx=*/3);
  auto linearEnc = LinearEncodingAttr::get(&ctx, ll);
  auto ty = RankedTensorType::get(shape, builder->getF32Type(), linearEnc);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(0u, 1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, NestedSliceUnwrap) {
  // SliceEncodingAttr has its own toLinearLayout (upstream
  // LinearLayoutConversions.cpp:1017) that produces the correct
  // sliced-rank LL. The analysis must NOT manually unwrap the parent
  // (doing so would apply the parent's LL at the wrong rank).
  //
  // Parent 3D: warpsPerCTA=[2,2,1]. Dim 2 has warp stride 1 → warp-invariant.
  // After slicing dim 2: rank-2 tensor, remaining dims inherit the parent's
  // warp basis minus the sliced dim. After slicing dim 1: rank-1, only dim 0
  // remains (originally dim 0 of parent), which has warpsPerCTA[0]=2 →
  // warp-partitioned → expected axis set is {}.
  SmallVector<unsigned> sizePerThread = {1, 1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 1, 32};
  SmallVector<unsigned> warpsPerCTA = {2, 2, 1};
  SmallVector<unsigned> order = {2, 1, 0};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/3);
  auto blocked3D = BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                            warpsPerCTA, order, cgaLayout);

  auto slice1 = SliceEncodingAttr::get(&ctx, /*dim=*/2, blocked3D);
  auto slice2 = SliceEncodingAttr::get(&ctx, /*dim=*/1, slice1);

  auto tyNested = RankedTensorType::get({64}, builder->getF32Type(), slice2);
  auto ty3D =
      RankedTensorType::get({2, 2, 64}, builder->getF32Type(), blocked3D);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  // 3D: dim 2 has warpsPerCTA==1 → warp-invariant → {2}.
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty3D), ElementsAre(2u));
  // 1D nested: only remaining dim (parent dim 0) has warpsPerCTA==2 →
  // warp-partitioned → {}.
  EXPECT_THAT(analysis.getWarpInvariantOutDims(tyNested), IsEmpty());
  EXPECT_FALSE(analysis.hasCrossSubgroupReuse(tyNested));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty3D));
}

TEST_F(SpatialReuseAnalysisTest, ScalarPointerLoad) {
  // Scalar pointer load (no tensor): the bool wrapper returns true
  // conservatively, while the axis-set entry point returns empty
  // because there are no tensor axes to report.
  auto module = buildModule();
  auto funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type i32Ty = builder->getI32Type();
  auto ptrTy = PointerType::get(i32Ty, /*addressSpace=*/1);

  funcOp.setType(builder->getFunctionType({ptrTy}, {}));
  Block *block = &funcOp.getBody().front();
  block->addArgument(ptrTy, builder->getUnknownLoc());
  BlockArgument ptr = block->getArgument(0);

  auto loadOp = LoadOp::create(*builder, builder->getUnknownLoc(), ptr);

  SpatialReuseAnalysis analysis(module);
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(loadOp));
  EXPECT_THAT(analysis.getWarpInvariantOutDims(loadOp), IsEmpty());
}

TEST_F(SpatialReuseAnalysisTest, NullptrEncoding) {
  // Tensor with nullptr encoding → conservative default returns the
  // full axis set {0, 1}. hasCrossSubgroupReuse: true.
  auto ty = RankedTensorType::get({32, 32}, builder->getI32Type(),
                                  /*encoding=*/Attribute{});

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(0u, 1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, NonPowerOfTwoShape) {
  // Non-power-of-2 dim (24) → conservative default returns full axis
  // set {0, 1}. hasCrossSubgroupReuse: true.
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 32};
  SmallVector<unsigned> warpsPerCTA = {1, 1};
  SmallVector<unsigned> order = {1, 0};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked = BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                          warpsPerCTA, order, cgaLayout);
  auto ty = RankedTensorType::get({24, 32}, builder->getF32Type(), blocked);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(0u, 1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, Subgroup2DBlockEncoding) {
  // Subgroup2DBlockEncodingAttr is only produced post-MaterializeBlockPointer.
  // This verifies pipeline-position independence.
  // Observed axis set: {1}. The LinearLayout produced by
  // subgroup2DBlockToLinearLayout has warp basis zero along dim 1 for this
  // (shape, instrShape, warpsPerCTA) configuration; warps distribute along
  // dim 0 only.
  DpasEncodingAttr dpas =
      makeDpas({2, 2}, /*repeatCount=*/8, /*systolicDepth=*/8,
               /*executionSize=*/16, /*opsPerChannel=*/2,
               /*repCluster=*/{1, 1}, /*threadsPerWarp=*/16);

  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> instrShape = {8, 16};
  unsigned numBlocks = 1;
  SmallVector<unsigned> order = {1, 0};
  unsigned kWidth = 2;

  auto subgroup2DBlock = Subgroup2DBlockEncodingAttr::get(
      &ctx, warpsPerCTA, dpas.getCGALayout(), instrShape, numBlocks, order,
      kWidth, dpas.getThreadsPerWarp());

  auto ty =
      RankedTensorType::get({32, 64}, builder->getF16Type(), subgroup2DBlock);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

TEST_F(SpatialReuseAnalysisTest, DescriptorLoadOpOverload) {
  // DescriptorLoadOp producing DotOperand{Dpas, opIdx=0}: op overload
  // forwards to type-based query. Expected axis set: {1} (K is
  // warp-broadcast). hasCrossSubgroupReuse: true.
  auto module = buildModule();
  auto funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  DpasEncodingAttr dpas = makeDpas();
  auto dotOpEnc =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  auto resultTy =
      RankedTensorType::get({64, 64}, builder->getF32Type(), dotOpEnc);

  Type elemTy = builder->getF32Type();
  ArrayRef<int64_t> shape = {64, 64};
  auto descTy =
      TensorDescType::get(shape, elemTy, /*sharedLayout=*/Attribute{});

  funcOp.setType(builder->getFunctionType({descTy}, {}));
  Block *block = &funcOp.getBody().front();
  block->addArgument(descTy, builder->getUnknownLoc());
  BlockArgument desc = block->getArgument(0);

  auto loadOp = DescriptorLoadOp::create(*builder, builder->getUnknownLoc(),
                                         resultTy, desc);

  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(loadOp), ElementsAre(1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(loadOp));
}

TEST_F(SpatialReuseAnalysisTest, DescriptorGatherOpOverload) {
  // DescriptorGatherOp producing Blocked-encoded tensor. Op overload
  // must match type-based query on result type.
  auto module = buildModule();
  auto funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked = BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                          warpsPerCTA, order, cgaLayout);
  auto resultTy =
      RankedTensorType::get({16, 32}, builder->getF16Type(), blocked);

  Type elemTy = builder->getF16Type();
  ArrayRef<int64_t> descShape = {1024, 1024};
  auto descTy =
      TensorDescType::get(descShape, elemTy, /*sharedLayout=*/Attribute{});
  auto indicesXTy = RankedTensorType::get({512}, builder->getI32Type());
  Type indicesYTy = builder->getI32Type();

  funcOp.setType(
      builder->getFunctionType({descTy, indicesXTy, indicesYTy}, {}));
  Block *block = &funcOp.getBody().front();
  block->addArgument(descTy, builder->getUnknownLoc());
  block->addArgument(indicesXTy, builder->getUnknownLoc());
  block->addArgument(indicesYTy, builder->getUnknownLoc());
  BlockArgument desc = block->getArgument(0);
  BlockArgument xOffsets = block->getArgument(1);
  BlockArgument yOffset = block->getArgument(2);

  auto gatherOp = DescriptorGatherOp::create(*builder, builder->getUnknownLoc(),
                                             resultTy, desc, xOffsets, yOffset);

  SpatialReuseAnalysis analysis(module);
  SmallVector<unsigned> opAxes = analysis.getWarpInvariantOutDims(gatherOp);
  SmallVector<unsigned> typeAxes = analysis.getWarpInvariantOutDims(resultTy);
  EXPECT_EQ(opAxes, typeAxes);
  EXPECT_EQ(analysis.hasCrossSubgroupReuse(gatherOp),
            analysis.hasCrossSubgroupReuse(resultTy));
}

TEST_F(SpatialReuseAnalysisTest, BareRankedTensorType) {
  // Same config as BlockedWarpsPartition via bare type query (no op).
  // Proves primitive is not op-bound.
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 32};
  SmallVector<unsigned> warpsPerCTA = {4, 1};
  SmallVector<unsigned> order = {1, 0};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked = BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                          warpsPerCTA, order, cgaLayout);
  auto ty = RankedTensorType::get({32, 32}, builder->getF32Type(), blocked);

  ModuleOp module = buildModule();
  SpatialReuseAnalysis analysis(module);
  EXPECT_THAT(analysis.getWarpInvariantOutDims(ty), ElementsAre(1u));
  EXPECT_TRUE(analysis.hasCrossSubgroupReuse(ty));
}

} // namespace
