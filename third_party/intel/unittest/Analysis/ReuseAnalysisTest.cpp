#include "intel/include/Analysis/ReuseAnalysis.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
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

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
namespace tti = mlir::triton::intel;

namespace {

class ReuseAnalysisTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<tt::TritonDialect>();
    ctx.getOrLoadDialect<ttg::TritonGPUDialect>();
    ctx.getOrLoadDialect<ttgi::TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  ModuleOp createModule() {
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    module->setAttr(ttg::AttrNumWarpsName, builder->getI32IntegerAttr(4));
    module->setAttr(ttg::AttrNumThreadsPerWarp, builder->getI32IntegerAttr(16));
    return module;
  }

  tt::FuncOp createFunction(ModuleOp module, StringRef name,
                            ArrayRef<Type> argTypes) {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToEnd(module.getBody());

    auto funcType = builder->getFunctionType(argTypes, {});
    auto funcOp =
        tt::FuncOp::create(*builder, builder->getUnknownLoc(), name, funcType);
    funcOp.addEntryBlock();
    builder->setInsertionPointToStart(&funcOp.getBody().front());

    return funcOp;
  }

  /// Like createFunction, but appends an empty `tt.return` at the end of
  /// the entry block and leaves the insertion point immediately BEFORE
  /// the return so the body can be built without caring about the
  /// terminator.
  tt::FuncOp createFunctionWithReturn(ModuleOp module, StringRef name,
                                      ArrayRef<Type> argTypes) {
    auto funcOp = createFunction(module, name, argTypes);
    // createFunction restores the insertion point on exit (via its
    // InsertionGuard), so we must explicitly re-enter the function body
    // before creating the terminator.
    builder->setInsertionPointToStart(&funcOp.getBody().front());
    auto loc = builder->getUnknownLoc();
    auto ret = tt::ReturnOp::create(*builder, loc, ValueRange{});
    builder->setInsertionPoint(ret);
    return funcOp;
  }

  Type getPtrType(Type elemType) { return tt::PointerType::get(elemType, 1); }

  /// Builds a BlockedEncodingAttr with a 1-CTA layout.
  ttg::BlockedEncodingAttr makeBlocked(ArrayRef<unsigned> sizePerThread,
                                       ArrayRef<unsigned> threadsPerWarp,
                                       ArrayRef<unsigned> warpsPerCTA,
                                       ArrayRef<unsigned> order) {
    auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(&ctx, order.size());
    return ttg::BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                         warpsPerCTA, order, cgaLayout);
  }

  /// Builds a DpasEncodingAttr (for spatial reuse tests).
  ttgi::DpasEncodingAttr
  makeDpas(unsigned repeatCount, unsigned systolicDepth, unsigned executionSize,
           unsigned opsPerChannel, ArrayRef<unsigned> warpsPerCTA,
           ArrayRef<unsigned> repCluster, unsigned threadsPerWarp) {
    return ttgi::DpasEncodingAttr::get(
        &ctx, repeatCount, systolicDepth, executionSize, opsPerChannel,
        warpsPerCTA, repCluster, threadsPerWarp, std::nullopt);
  }

  /// Creates a tt.load with default cache/eviction attributes.
  tt::LoadOp makeLoad(Value ptr, Type resultType) {
    return tt::LoadOp::create(
        *builder, builder->getUnknownLoc(), resultType, ptr, /*mask=*/Value(),
        /*other=*/Value(), tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// ===----------------------------------------------------------------------===//
// Test Reuse Facade
// ===----------------------------------------------------------------------===//

// Spatial analysis reports reuse (warp-invariant dim) while temporal reports
// none (load outside any loop). Union: anyReuse == true.
TEST_F(ReuseAnalysisTest, UnionTrueFromSpatial) {
  auto module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  // DPAS dot-operand A with warpsPerCTA=[4,1]: warps tile M (dim 0), K (dim 1)
  // is warp-invariant -> SpatialReuseAnalysis reports reuse.
  auto dpasEnc = makeDpas(/*repeatCount=*/8, /*systolicDepth=*/8,
                          /*executionSize=*/16, /*opsPerChannel=*/2,
                          /*warpsPerCTA=*/{4, 1}, /*repCluster=*/{1, 1},
                          /*threadsPerWarp=*/16);
  auto dotOpEnc = ttg::DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpasEnc,
                                                   /*kWidth=*/1);
  auto resultTy = RankedTensorType::get({8, 16}, f16Ty, dotOpEnc);
  auto ptrTensorTy = RankedTensorType::get({8, 16}, ptrType, dotOpEnc);

  // Splat pointer and load OUTSIDE any loop (TemporalReuseAnalysis -> false).
  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);
  auto loadOp = makeLoad(splatPtr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(module);
  tti::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  ttgi::ReuseAnalysis analysis(module, strideAnalysis);

  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

// Spatial analysis reports no reuse (1-D tensor with warps tiling the only
// axis) while temporal reports reuse (loop-invariant pointer).
// Union: anyReuse == true.
TEST_F(ReuseAnalysisTest, UnionTrueFromTemporal) {
  auto module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  // 1-D blocked encoding with warps distributed along the only axis:
  // SpatialReuseAnalysis -> no warp-invariant dim.
  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, ptrType, blocked);

  // Splat pointer BEFORE the loop so it's loop-invariant inside the loop body.
  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);

  // Build scf.for and place the load inside with the loop-invariant pointer.
  auto i32Type = builder->getI32Type();
  auto lb = arith::ConstantOp::create(*builder, loc, i32Type,
                                      builder->getI32IntegerAttr(0));
  auto ub = arith::ConstantOp::create(*builder, loc, i32Type,
                                      builder->getI32IntegerAttr(10));
  auto step = arith::ConstantOp::create(*builder, loc, i32Type,
                                        builder->getI32IntegerAttr(1));
  auto forOp = scf::ForOp::create(*builder, loc, lb, ub, step);

  tt::LoadOp loadOp;
  {
    OpBuilder::InsertionGuard loopGuard(*builder);
    builder->setInsertionPointToStart(forOp.getBody());
    loadOp = makeLoad(splatPtr, resultTy);
  }

  tti::ModuleAxisInfoAnalysis axisInfo(module);
  tti::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  ttgi::ReuseAnalysis analysis(module, strideAnalysis);

  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

// Neither analysis reports reuse: 1-D streaming load outside any loop with
// warps tiling the only axis. Union: anyReuse == false.
TEST_F(ReuseAnalysisTest, UnionFalse) {
  auto module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, ptrType, blocked);

  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);
  auto loadOp = makeLoad(splatPtr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(module);
  tti::ModuleStrideAnalysis strideAnalysis(module, axisInfo);
  ttgi::ReuseAnalysis analysis(module, strideAnalysis);

  EXPECT_FALSE(analysis.anyReuse(loadOp));
}

} // namespace
