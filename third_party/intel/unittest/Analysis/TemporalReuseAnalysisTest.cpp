#include "intel/include/Analysis/TemporalReuseAnalysis.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
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
using namespace mlir::triton::gpu::intel;

namespace {

class TemporalReuseAnalysisTest : public ::testing::Test {
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
    funcOp.addEntryBlock();
    return funcOp;
  }

  // SCF loop builders for temporal reuse tests
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

  scf::WhileOp buildScfWhile(OpBuilder &b, Type ivType, Value initValue) {
    Location loc = b.getUnknownLoc();
    auto whileOp =
        scf::WhileOp::create(b, loc, TypeRange(ivType), ValueRange(initValue));

    // Before region: condition check
    Block *beforeBlock =
        b.createBlock(&whileOp.getBefore(), {}, {ivType}, {loc});
    b.setInsertionPointToStart(beforeBlock);
    auto cond = arith::ConstantIntOp::create(b, loc, b.getI1Type(), 1);
    scf::ConditionOp::create(b, loc, cond, beforeBlock->getArguments());

    // After region: body
    Block *afterBlock = b.createBlock(&whileOp.getAfter(), {}, {ivType}, {loc});
    b.setInsertionPointToStart(afterBlock);

    return whileOp;
  }

  // Helper to build a BlockedEncodingAttr
  BlockedEncodingAttr makeBlocked(ArrayRef<unsigned> sizePerThread,
                                  ArrayRef<unsigned> threadsPerWarp,
                                  ArrayRef<unsigned> warpsPerCTA,
                                  ArrayRef<unsigned> order) {
    auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, order.size());
    return BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                    warpsPerCTA, order, cgaLayout);
  }

  // Helper to build a tt.load with a pointer operand and tensor result type
  LoadOp buildLoadOp(OpBuilder &b, Value ptr, RankedTensorType resultTy) {
    Location loc = b.getUnknownLoc();
    return LoadOp::create(b, loc, resultTy, ptr);
  }

  // Helper to build a tt.descriptor_load
  DescriptorLoadOp buildDescriptorLoadOp(OpBuilder &b, Value desc,
                                         RankedTensorType resultTy) {
    Location loc = b.getUnknownLoc();
    return DescriptorLoadOp::create(b, loc, resultTy, desc);
  }

  // Helper to build a tt.descriptor_load with explicit indices
  DescriptorLoadOp buildDescriptorLoadOp(OpBuilder &b, Value desc,
                                         ValueRange indices,
                                         RankedTensorType resultTy) {
    Location loc = b.getUnknownLoc();
    return DescriptorLoadOp::create(b, loc, resultTy, desc, indices);
  }

  // Helper to build a tt.descriptor_gather
  DescriptorGatherOp buildDescriptorGatherOp(OpBuilder &b, Value desc,
                                             Value xOffsets, Value yOffset,
                                             RankedTensorType resultTy) {
    Location loc = b.getUnknownLoc();
    return DescriptorGatherOp::create(b, loc, resultTy, desc, xOffsets,
                                      yOffset);
  }

  // Helper to add a pointer argument to a function
  BlockArgument addPtrArg(func::FuncOp funcOp, Type elemTy,
                          unsigned addressSpace = 1) {
    Block *block = &funcOp.getBody().front();
    auto ptrTy = PointerType::get(elemTy, addressSpace);
    FunctionType newFuncType = builder->getFunctionType(
        {ptrTy}, funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
    return block->addArgument(ptrTy, builder->getUnknownLoc());
  }

  // Helper to add a tensor_desc argument to a function
  BlockArgument addDescArg(func::FuncOp funcOp, ArrayRef<int64_t> shape,
                           Type elemTy) {
    Block *block = &funcOp.getBody().front();
    auto descTy =
        TensorDescType::get(shape, elemTy, /*sharedLayout=*/Attribute{});
    FunctionType newFuncType = builder->getFunctionType(
        {descTy}, funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
    return block->addArgument(descTy, builder->getUnknownLoc());
  }

  // Helper to build a tensor of pointers type
  RankedTensorType makePtrTensorType(ArrayRef<int64_t> shape, Type elemTy,
                                     Attribute encoding,
                                     unsigned addressSpace = 1) {
    auto ptrTy = PointerType::get(elemTy, addressSpace);
    return RankedTensorType::get(shape, ptrTy, encoding);
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// Test case 1: Load with loop-invariant pointer inside scf.for
// Expected: hasTemporalReuse == true (Case A)
TEST_F(TemporalReuseAnalysisTest, LoopInvariantPointer) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  // Add pointer argument
  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build blocked encoding for result
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);

  // Splat the pointer to match result shape.  The splat must be created
  // *before* entering the loop body so that `isDefinedOutsideOfLoop`
  // returns true (Case A).  Creating it inside the body would force the
  // analysis down the Case B "partial-axis" path instead.
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Build scf.for (builder insertion point moves into the loop body).
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);

  // Build load inside loop with loop-invariant pointer
  LoadOp loadOp = buildLoadOp(*builder, splatPtr, resultTy);

  // Yield
  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Case A: loop-invariant
}

// Test case 2: Load inside scf.for with pointer = basePtr + IV*N streaming
// along the sole tensor axis. Every axis has positive IV-stride (Case C),
// so the load has no temporal reuse.  Expected: hasTemporalReuse == false.
TEST_F(TemporalReuseAnalysisTest, StreamingPointerOneDim) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build scf.for
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1};
  SmallVector<unsigned> threadsPerWarp = {32};
  SmallVector<unsigned> warpsPerCTA = {4};
  SmallVector<unsigned> order = {0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);

  // Splat base pointer
  RankedTensorType ptrTensorTy = makePtrTensorType({128}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Build stride = IV * 128
  auto c128 =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 128);
  auto stride =
      arith::MulIOp::create(*builder, builder->getUnknownLoc(), iv, c128);

  // Splat stride
  Type i64Ty = builder->getI64Type();
  auto strideTensorTy = RankedTensorType::get({128}, i64Ty, blocked);
  auto strideI64 = arith::IndexCastOp::create(
      *builder, builder->getUnknownLoc(), i64Ty, stride);
  auto splatStride = SplatOp::create(*builder, builder->getUnknownLoc(),
                                     strideTensorTy, strideI64);

  // AddPtr: ptr = basePtr + IV*128
  auto advancedPtr = AddPtrOp::create(*builder, builder->getUnknownLoc(),
                                      ptrTensorTy, splatPtr, splatStride);

  // Load
  LoadOp loadOp = buildLoadOp(*builder, advancedPtr, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_FALSE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_FALSE(reuseByDepth[0]); // Case C: pure streaming, no temporal reuse
}

// Test case 3: Load inside scf.for with pointer advancing along K axis only;
// tensor has M and K axes. Expected: hasTemporalReuse == true (Case B — GEMM
// A-operand)
TEST_F(TemporalReuseAnalysisTest, PartialAxisStreaming) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build scf.for (K loop)
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // Build K-only IV-dependent offset via expand_dims + broadcast
  // so IV-stride is [0, 32] (M axis stays, K axis advances).
  auto c32 =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 32);
  auto kStride =
      arith::MulIOp::create(*builder, builder->getUnknownLoc(), iv, c32);

  Type i64Ty = builder->getI64Type();
  auto kStrideI64 = arith::IndexCastOp::create(
      *builder, builder->getUnknownLoc(), i64Ty, kStride);

  // Splat to 1D K-only tensor<32xi64>
  BlockedEncodingAttr blocked1D = makeBlocked({1}, {32}, {4}, {0});
  auto ty1D = RankedTensorType::get({32}, i64Ty, blocked1D);
  auto splat1D =
      SplatOp::create(*builder, builder->getUnknownLoc(), ty1D, kStrideI64);

  // expand_dims axis=0 -> tensor<1x32xi64>
  BlockedEncodingAttr blocked2D_1x32 =
      makeBlocked({1, 1}, {1, 32}, {1, 4}, {1, 0});
  auto ty1x32 = RankedTensorType::get({1, 32}, i64Ty, blocked2D_1x32);
  auto expanded = triton::ExpandDimsOp::create(
      *builder, builder->getUnknownLoc(), ty1x32, splat1D, 0);

  // broadcast to tensor<16x32xi64>
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto ty16x32 = RankedTensorType::get({16, 32}, i64Ty, blocked);
  auto bcast = triton::BroadcastOp::create(*builder, builder->getUnknownLoc(),
                                           ty16x32, expanded);

  // Splat base pointer and addptr
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);
  auto advancedPtr = AddPtrOp::create(*builder, builder->getUnknownLoc(),
                                      ptrTensorTy, splatPtr, bcast);

  // Load
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);
  LoadOp loadOp = buildLoadOp(*builder, advancedPtr, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Case B: K advances but M stays fixed
}

// Test case 4: Load outside any loop
// Expected: hasTemporalReuse == false; getReuseByLoopDepth == {}
TEST_F(TemporalReuseAnalysisTest, LoadOutsideLoop) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);

  // Splat pointer
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Load outside any loop
  LoadOp loadOp = buildLoadOp(*builder, splatPtr, resultTy);

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_FALSE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  EXPECT_EQ(reuseByDepth.size(), 0u); // Empty: not in any loop
}

// Test case 5: Nested scf.for: inner streaming, outer loop-invariant
// Expected: hasTemporalReuse == true (outer Case A wins)
TEST_F(TemporalReuseAnalysisTest, NestedLoopOuterInvariant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Outer loop
  scf::ForOp outerFor = buildScfFor(*builder, 0, 10, 1);

  // Inner loop
  scf::ForOp innerFor = buildScfFor(*builder, 0, 10, 1);
  Value innerIV = innerFor.getInductionVar();

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1};
  SmallVector<unsigned> threadsPerWarp = {32};
  SmallVector<unsigned> warpsPerCTA = {4};
  SmallVector<unsigned> order = {0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);

  // Splat base pointer
  RankedTensorType ptrTensorTy = makePtrTensorType({128}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Build stride = innerIV * 128 (streaming in inner loop)
  auto c128 =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 128);
  auto stride =
      arith::MulIOp::create(*builder, builder->getUnknownLoc(), innerIV, c128);

  Type i64Ty = builder->getI64Type();
  auto strideTensorTy = RankedTensorType::get({128}, i64Ty, blocked);
  auto strideI64 = arith::IndexCastOp::create(
      *builder, builder->getUnknownLoc(), i64Ty, stride);
  auto splatStride = SplatOp::create(*builder, builder->getUnknownLoc(),
                                     strideTensorTy, strideI64);

  // AddPtr
  auto advancedPtr = AddPtrOp::create(*builder, builder->getUnknownLoc(),
                                      ptrTensorTy, splatPtr, splatStride);

  // Load
  LoadOp loadOp = buildLoadOp(*builder, advancedPtr, resultTy);

  // Inner yield
  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Outer yield
  builder->setInsertionPointAfter(innerFor);
  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 2u);
  EXPECT_FALSE(reuseByDepth[0]); // Inner loop: streaming (Case C)
  EXPECT_TRUE(reuseByDepth[1]);  // Outer loop: loop-invariant (Case A)
}

// Test case 6: Load whose pointer is carried through a loop as an iter-arg
// without being tracked by the dataflow solver.
// Expected: hasTemporalReuse == true (conservative default)
TEST_F(TemporalReuseAnalysisTest, LoopCarriedUntrackedPointer) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto initPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Build scf.for carrying the pointer as an iter-arg
  auto lb =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 0);
  auto ub =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 10);
  auto step =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 1);
  auto forOp = scf::ForOp::create(*builder, builder->getUnknownLoc(), lb, ub,
                                  step, ValueRange{initPtr});

  builder->setInsertionPointToStart(forOp.getBody());
  Value carriedPtr = forOp.getRegionIterArgs()[0];

  // Load from carried pointer
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);
  LoadOp loadOp = buildLoadOp(*builder, carriedPtr, resultTy);

  // Yield unchanged pointer
  scf::YieldOp::create(*builder, builder->getUnknownLoc(),
                       ValueRange{carriedPtr});

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Conservative default
}

// Test case 7: Load inside scf.while whose pointer is the `after` region's
// block argument.  `isDefinedOutsideOfLoop` returns false for the region
// argument, and StrideInfo does not track scf.while region args, so the
// classifier falls into the Unknown -> Held conservative path and reports
// reuse.  Expected: hasTemporalReuse == true (conservative fallback, not
// Case A).
TEST_F(TemporalReuseAnalysisTest, WhileLoopInvariant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);

  // Splat base pointer
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Build scf.while carrying the pointer
  scf::WhileOp whileOp = buildScfWhile(*builder, ptrTensorTy, splatPtr);

  // Move insertion point to after block (body of the while)
  Block *afterBlock = &whileOp.getAfter().front();
  builder->setInsertionPointToStart(afterBlock);
  Value carriedPtr = afterBlock->getArgument(0);

  // Load using carried pointer (unchanged)
  LoadOp loadOp = buildLoadOp(*builder, carriedPtr, resultTy);

  // Yield unchanged pointer
  scf::YieldOp::create(*builder, builder->getUnknownLoc(),
                       ValueRange(carriedPtr));

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Conservative fallback (StrideInfo does
                                // not track scf.while region args)
}

// Test case 8: tt.descriptor_load inside scf.for with a loop-invariant
// descriptor (Case A, since the descriptor is a function argument).
// Expected: hasTemporalReuse == true (validates DescriptorLoadOp overload).
TEST_F(TemporalReuseAnalysisTest, DescriptorLoadInvariant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  SmallVector<int64_t> descShape = {1024, 1024};
  BlockArgument desc = addDescArg(funcOp, descShape, f16Ty);

  // Build scf.for
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);

  // The result encoding is irrelevant to the analysis — it only inspects the
  // descriptor operand's stride — so any distributed encoding works.
  BlockedEncodingAttr blocked = makeBlocked(/*sizePerThread=*/{1, 1},
                                            /*threadsPerWarp=*/{1, 16},
                                            /*warpsPerCTA=*/{4, 1},
                                            /*order=*/{1, 0});
  auto resultTy = RankedTensorType::get({64, 64}, f16Ty, blocked);

  // Descriptor load
  DescriptorLoadOp loadOp = buildDescriptorLoadOp(*builder, desc, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Case A: loop-invariant descriptor
}

// Test case 9: tt.descriptor_gather inside scf.for with Case A loop-invariant
// pointer. Expected: hasTemporalReuse == true (validates DescriptorGatherOp
// overload)
TEST_F(TemporalReuseAnalysisTest, DescriptorGatherInvariant) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  // Per tt.descriptor_gather spec, the descriptor block must have 1 row.
  SmallVector<int64_t> descShape = {1, 32};
  BlockArgument desc = addDescArg(funcOp, descShape, f16Ty);

  // Build scf.for
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);

  // Build blocked encoding for the 2D gather result (rows x cols).
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {2, 16};
  SmallVector<unsigned> warpsPerCTA = {2, 2};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  // Result shape is (indices_len, desc_col_width) = (16, 32).
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);

  // Per spec: indices are a 1D tensor, length matches result row count.
  auto indicesXTy = RankedTensorType::get({16}, builder->getI32Type());
  Type indicesYTy = builder->getI32Type();

  // Add indices as function arguments (loop-invariant). Update the function
  // type and the block argument list atomically via `funcOp.insertArgument`
  // to avoid a transient mismatch between signature and block.
  Block *block = &funcOp.getBody().front();
  unsigned xArgIdx = block->getNumArguments();
  (void)funcOp.insertArgument(xArgIdx, indicesXTy, /*argAttrs=*/{},
                              builder->getUnknownLoc());
  unsigned yArgIdx = block->getNumArguments();
  (void)funcOp.insertArgument(yArgIdx, indicesYTy, /*argAttrs=*/{},
                              builder->getUnknownLoc());
  Value xOffsets = block->getArgument(xArgIdx);
  Value yOffset = block->getArgument(yArgIdx);

  // Descriptor gather
  DescriptorGatherOp gatherOp =
      buildDescriptorGatherOp(*builder, desc, xOffsets, yOffset, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(gatherOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(gatherOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Case A: loop-invariant descriptor
}

// tt.descriptor_load with a loop-invariant descriptor but an IV-derived
// index operand.  Under the old (desc-only) classifier this reported
// reuse because the descriptor is a function argument and lands in
// Case A.  The corrected classifier also inspects `indices`, sees the
// streaming IV-cast index, and reports no reuse.
// Expected: hasTemporalReuse == false.
TEST_F(TemporalReuseAnalysisTest, DescriptorLoadStreamingIndex) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  SmallVector<int64_t> descShape = {1024, 1024};
  BlockArgument desc = addDescArg(funcOp, descShape, f16Ty);

  // Build scf.for (K loop); iv lives inside the loop body.
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // Invariant x index (constant 0) and streaming y index (cast from iv).
  Location loc = builder->getUnknownLoc();
  auto xIdx =
      arith::ConstantIntOp::create(*builder, loc, builder->getI32Type(), 0);
  auto yIdx =
      arith::IndexCastOp::create(*builder, loc, builder->getI32Type(), iv);

  BlockedEncodingAttr blocked = makeBlocked(/*sizePerThread=*/{1, 1},
                                            /*threadsPerWarp=*/{1, 16},
                                            /*warpsPerCTA=*/{4, 1},
                                            /*order=*/{1, 0});
  auto resultTy = RankedTensorType::get({64, 64}, f16Ty, blocked);

  DescriptorLoadOp loadOp =
      buildDescriptorLoadOp(*builder, desc, ValueRange{xIdx, yIdx}, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_FALSE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_FALSE(reuseByDepth[0]); // yIdx streams -> no reuse
}

// tt.descriptor_gather with a loop-invariant descriptor and xOffsets but
// an IV-derived scalar y_offset.  Mirrors DescriptorLoadStreamingIndex
// for the gather op: exercises the multi-operand classification path.
// Expected: hasTemporalReuse == false.
TEST_F(TemporalReuseAnalysisTest, DescriptorGatherStreamingYOffset) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  // Per tt.descriptor_gather spec, the descriptor block must have 1 row.
  SmallVector<int64_t> descShape = {1, 32};
  BlockArgument desc = addDescArg(funcOp, descShape, f16Ty);

  // Invariant xOffsets (func arg); y_offset will be built inside the loop.
  auto indicesXTy = RankedTensorType::get({16}, builder->getI32Type());
  Block *block = &funcOp.getBody().front();
  unsigned xArgIdx = block->getNumArguments();
  (void)funcOp.insertArgument(xArgIdx, indicesXTy, /*argAttrs=*/{},
                              builder->getUnknownLoc());
  Value xOffsets = block->getArgument(xArgIdx);

  // Build scf.for and derive yOffset from the IV.
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();
  Location loc = builder->getUnknownLoc();
  auto yOffset =
      arith::IndexCastOp::create(*builder, loc, builder->getI32Type(), iv);

  // Blocked encoding for the 2D gather result (rows x cols).
  BlockedEncodingAttr blocked = makeBlocked(/*sizePerThread=*/{1, 1},
                                            /*threadsPerWarp=*/{2, 16},
                                            /*warpsPerCTA=*/{2, 2},
                                            /*order=*/{1, 0});
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);

  DescriptorGatherOp gatherOp =
      buildDescriptorGatherOp(*builder, desc, xOffsets, yOffset, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_FALSE(analysis.hasTemporalReuse(gatherOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(gatherOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_FALSE(reuseByDepth[0]); // yOffset streams -> no reuse
}

// Test case 10: Load inside two-deep scf.for nest where outer loop is Case A
// (loop-invariant) and inner loop was Case C (streaming), now collapsed.
// Expected: getReuseByLoopDepth == {true, true} (innermost first, outermost
// last) and hasTemporalReuse == true. Exercises the structural query.
TEST_F(TemporalReuseAnalysisTest, TwoDeepNestStructuralQuery) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build blocked encoding
  SmallVector<unsigned> sizePerThread = {1};
  SmallVector<unsigned> threadsPerWarp = {32};
  SmallVector<unsigned> warpsPerCTA = {4};
  SmallVector<unsigned> order = {0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);

  // Splat base pointer (loop-invariant for outer)
  RankedTensorType ptrTensorTy = makePtrTensorType({128}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);

  // Outer loop (Case A: loop-invariant)
  scf::ForOp outerFor = buildScfFor(*builder, 0, 10, 1);

  // Inner loop (Case C: streaming)
  scf::ForOp innerFor = buildScfFor(*builder, 0, 10, 1);
  Value innerIV = innerFor.getInductionVar();

  // Build stride = innerIV * 128 (streaming along sole axis in inner loop)
  auto c128 =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 128);
  auto stride =
      arith::MulIOp::create(*builder, builder->getUnknownLoc(), innerIV, c128);

  Type i64Ty = builder->getI64Type();
  auto strideTensorTy = RankedTensorType::get({128}, i64Ty, blocked);
  auto strideI64 = arith::IndexCastOp::create(
      *builder, builder->getUnknownLoc(), i64Ty, stride);
  auto splatStride = SplatOp::create(*builder, builder->getUnknownLoc(),
                                     strideTensorTy, strideI64);

  // AddPtr
  auto advancedPtr = AddPtrOp::create(*builder, builder->getUnknownLoc(),
                                      ptrTensorTy, splatPtr, splatStride);

  // Load
  LoadOp loadOp = buildLoadOp(*builder, advancedPtr, resultTy);

  // Inner yield
  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Outer yield
  builder->setInsertionPointAfter(innerFor);
  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 2u);
  EXPECT_FALSE(reuseByDepth[0]); // Innermost: streaming (Case C)
  EXPECT_TRUE(reuseByDepth[1]);  // Outermost: loop-invariant (Case A)
}

// Test case 11: GEMM A-operand pattern (2D tensor with K-loop advancing along K
// axis only, M axis stays constant). Expected: hasTemporalReuse == true, Case B
// partial-axis reuse (M axis IV-stride is 0).
TEST_F(TemporalReuseAnalysisTest, GEMMAOperandCaseB) {
  ModuleOp module = buildModule();
  func::FuncOp funcOp = buildFunc(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  Type f16Ty = builder->getF16Type();
  BlockArgument basePtr = addPtrArg(funcOp, f16Ty);

  // Build scf.for (K loop)
  scf::ForOp forOp = buildScfFor(*builder, 0, 10, 1);
  Value iv = forOp.getInductionVar();

  // Build K-only IV-dependent offset via expand_dims + broadcast
  // so IV-stride is [0, 32] (M axis stays, K axis advances).
  // Use different encoding parameters than PartialAxisStreaming to
  // exercise a different configuration.
  auto c32 =
      arith::ConstantIndexOp::create(*builder, builder->getUnknownLoc(), 32);
  auto kStride =
      arith::MulIOp::create(*builder, builder->getUnknownLoc(), iv, c32);

  Type i64Ty = builder->getI64Type();
  auto kStrideI64 = arith::IndexCastOp::create(
      *builder, builder->getUnknownLoc(), i64Ty, kStride);

  // Splat to 1D K-only tensor<32xi64>
  BlockedEncodingAttr blocked1D = makeBlocked({1}, {16}, {8}, {0});
  auto ty1D = RankedTensorType::get({32}, i64Ty, blocked1D);
  auto splat1D =
      SplatOp::create(*builder, builder->getUnknownLoc(), ty1D, kStrideI64);

  // expand_dims axis=0 -> tensor<1x32xi64>
  BlockedEncodingAttr blocked2D_1x32 =
      makeBlocked({1, 1}, {1, 16}, {1, 8}, {1, 0});
  auto ty1x32 = RankedTensorType::get({1, 32}, i64Ty, blocked2D_1x32);
  auto expanded = triton::ExpandDimsOp::create(
      *builder, builder->getUnknownLoc(), ty1x32, splat1D, 0);

  // broadcast to tensor<16x32xi64>
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 16};
  SmallVector<unsigned> warpsPerCTA = {4, 1};
  SmallVector<unsigned> order = {1, 0};
  BlockedEncodingAttr blocked =
      makeBlocked(sizePerThread, threadsPerWarp, warpsPerCTA, order);
  auto ty16x32 = RankedTensorType::get({16, 32}, i64Ty, blocked);
  auto bcast = triton::BroadcastOp::create(*builder, builder->getUnknownLoc(),
                                           ty16x32, expanded);

  // Splat base pointer and addptr
  RankedTensorType ptrTensorTy = makePtrTensorType({16, 32}, f16Ty, blocked);
  auto splatPtr =
      SplatOp::create(*builder, builder->getUnknownLoc(), ptrTensorTy, basePtr);
  auto advancedPtr = AddPtrOp::create(*builder, builder->getUnknownLoc(),
                                      ptrTensorTy, splatPtr, bcast);

  // Load
  auto resultTy = RankedTensorType::get({16, 32}, f16Ty, blocked);
  LoadOp loadOp = buildLoadOp(*builder, advancedPtr, resultTy);

  scf::YieldOp::create(*builder, builder->getUnknownLoc());

  // Analyze
  mlir::triton::intel::ModuleAxisInfoAnalysis axisInfo(module);
  mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  TemporalReuseAnalysis analysis(strideAnalysis);

  EXPECT_TRUE(analysis.hasTemporalReuse(loadOp));
  SmallVector<bool> reuseByDepth = analysis.getReuseByLoopDepth(loadOp);
  ASSERT_EQ(reuseByDepth.size(), 1u);
  EXPECT_TRUE(reuseByDepth[0]); // Case B: M axis IV-stride is 0
}

} // namespace
