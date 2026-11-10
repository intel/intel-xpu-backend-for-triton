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
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gmock/gmock.h>
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

  // Returns an owning handle so the caller controls the module's lifetime.
  OwningOpRef<ModuleOp> createModule() {
    auto loc = builder->getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    (*module)->setAttr(ttg::AttrNumWarpsName, builder->getI32IntegerAttr(4));
    (*module)->setAttr(ttg::AttrNumThreadsPerWarp,
                       builder->getI32IntegerAttr(16));
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

  Type getPtrType(Type elemType) { return tt::PointerType::get(elemType); }

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

  /// Creates a tt.load with the default cache policy.
  tt::LoadOp makeLoad(Value ptr, Type resultType) {
    return tt::LoadOp::create(*builder, builder->getUnknownLoc(), resultType,
                              ptr, /*mask=*/Value(), /*other=*/Value(),
                              /*cachePolicy=*/Attribute(),
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
  OwningOpRef<ModuleOp> module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
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

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

// Spatial analysis reports no reuse (1-D tensor with warps tiling the only
// axis) while temporal reports reuse (loop-invariant pointer).
// Union: anyReuse == true.
TEST_F(ReuseAnalysisTest, UnionTrueFromTemporal) {
  OwningOpRef<ModuleOp> module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
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

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

// Neither analysis reports reuse: 1-D streaming load outside any loop with
// warps tiling the only axis. Union: anyReuse == false.
TEST_F(ReuseAnalysisTest, UnionFalse) {
  OwningOpRef<ModuleOp> module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, ptrType, blocked);

  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);
  auto loadOp = makeLoad(splatPtr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_FALSE(analysis.anyReuse(loadOp));
}

// Same layout and loop-free placement as UnionFalse, but the offset comes from
// a tensor loaded by the kernel (`table[idx]`): reuse is data-dependent and
// invisible to both legs, so the suppress-side anyReuse is conservatively
// true. It is not a proof, so knownReuse stays false.
TEST_F(ReuseAnalysisTest, GatherAddressIsConservativeReuse) {
  OwningOpRef<ModuleOp> module = createModule();
  auto i32Ty = builder->getI32Type();
  auto ptrType = getPtrType(i32Ty);
  auto funcOp =
      createFunctionWithReturn(*module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();

  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto valTy = RankedTensorType::get({128}, i32Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, ptrType, blocked);

  auto range = tt::MakeRangeOp::create(*builder, loc, valTy, 0, 128);
  auto idxBase =
      tt::SplatOp::create(*builder, loc, ptrTensorTy, funcOp.getArgument(0));
  auto idxPtr = tt::AddPtrOp::create(*builder, loc, ptrTensorTy, idxBase,
                                     range.getResult());
  auto idxLoad = makeLoad(idxPtr, valTy);
  auto tableBase =
      tt::SplatOp::create(*builder, loc, ptrTensorTy, funcOp.getArgument(1));
  auto tablePtr = tt::AddPtrOp::create(*builder, loc, ptrTensorTy, tableBase,
                                       idxLoad.getResult());
  auto tableLoad = makeLoad(tablePtr, valTy);

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_TRUE(analysis.anyReuse(tableLoad));
  EXPECT_FALSE(analysis.knownReuse(tableLoad));
  // The index load itself is structured streaming.
  EXPECT_FALSE(analysis.anyReuse(idxLoad));
}

// A scalar load feeding the offset (e.g. a per-sequence start) is uniform
// across the tile, so the address stays structured: anyReuse == false.
TEST_F(ReuseAnalysisTest, ScalarLoadedOffsetIsNotReuse) {
  OwningOpRef<ModuleOp> module = createModule();
  auto i32Ty = builder->getI32Type();
  auto f16Ty = builder->getF16Type();
  auto funcOp = createFunctionWithReturn(
      *module, "test_func", {getPtrType(i32Ty), getPtrType(f16Ty)});
  auto loc = builder->getUnknownLoc();

  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto rangeTy = RankedTensorType::get({128}, i32Ty, blocked);
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, getPtrType(f16Ty), blocked);

  auto start = makeLoad(funcOp.getArgument(0), i32Ty);
  auto base = tt::AddPtrOp::create(*builder, loc, getPtrType(f16Ty),
                                   funcOp.getArgument(1), start.getResult());
  auto splat = tt::SplatOp::create(*builder, loc, ptrTensorTy, base);
  auto range = tt::MakeRangeOp::create(*builder, loc, rangeTy, 0, 128);
  auto ptr = tt::AddPtrOp::create(*builder, loc, ptrTensorTy, splat,
                                  range.getResult());
  auto loadOp = makeLoad(ptr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_FALSE(analysis.anyReuse(loadOp));
}

// ===----------------------------------------------------------------------===//
// addressDependsOnLoadedTensor: one case per branch of the def-chain walk.
// Each kernel tags the queried load with `{probe}`. The helper is queried
// directly because, inside loops, the temporal leg alone already makes
// anyReuse true and would hide a broken walk.
// ===----------------------------------------------------------------------===//

class AddressDependsOnLoadedTensorTest : public ReuseAnalysisTest {
public:
  /// Parses `body` as a single `tt.func` and returns the walk's verdict for
  /// the load tagged `{probe}`.
  bool probe(StringRef body) {
    std::string ir = R"(
#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
)" + body.str() + "\n}\n";
    module = parseSourceString<ModuleOp>(ir, &ctx);
    EXPECT_TRUE(module) << "failed to parse:\n" << ir;
    if (!module)
      return false;
    tt::LoadOp probed;
    module->walk([&](tt::LoadOp load) {
      if (load->hasAttr("probe"))
        probed = load;
    });
    EXPECT_TRUE(probed) << "no load tagged {probe}";
    return probed && ttgi::addressDependsOnLoadedTensor(probed.getPtr());
  }

private:
  OwningOpRef<ModuleOp> module;
};

// Gathered pointer formed outside the loop, reached via the iter_arg's init.
TEST_F(AddressDependsOnLoadedTensorTest, ForIterArgInit) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%idx: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>, %lb: i32, %ub: i32) {
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant dense<32> : tensor<32xi32, #b>
    %tok = tt.load %idx : tensor<32x!tt.ptr<i32>, #b>
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p0 = tt.addptr %st, %tok : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %r = scf.for %i = %lb to %ub step %c1 iter_args(%p = %p0) -> (tensor<32x!tt.ptr<i32>, #b>) : i32 {
      %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
      %pn = tt.addptr %p, %c32 : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
      scf.yield %pn : tensor<32x!tt.ptr<i32>, #b>
    }
    tt.return
  })"));
}

// Structured init; the gather enters only through the yielded value.
TEST_F(AddressDependsOnLoadedTensorTest, ForIterArgYield) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%idx: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>, %lb: i32, %ub: i32) {
    %c1 = arith.constant 1 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p0 = tt.addptr %st, %range : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %r = scf.for %i = %lb to %ub step %c1 iter_args(%p = %p0) -> (tensor<32x!tt.ptr<i32>, #b>) : i32 {
      %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
      %tok = tt.load %idx : tensor<32x!tt.ptr<i32>, #b>
      %pn = tt.addptr %p, %tok : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
      scf.yield %pn : tensor<32x!tt.ptr<i32>, #b>
    }
    tt.return
  })"));
}

// The offset is an scf.for result whose yielded value is a loaded tensor.
TEST_F(AddressDependsOnLoadedTensorTest, ForResult) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%idx: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>, %lb: i32, %ub: i32) {
    %c1 = arith.constant 1 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %off = scf.for %i = %lb to %ub step %c1 iter_args(%o = %range) -> (tensor<32xi32, #b>) : i32 {
      %tok = tt.load %idx : tensor<32x!tt.ptr<i32>, #b>
      scf.yield %tok : tensor<32xi32, #b>
    }
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p = tt.addptr %st, %off : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
    tt.return
  })"));
}

// The offset is an scf.if result; the gather is only on the else branch.
TEST_F(AddressDependsOnLoadedTensorTest, IfElseResult) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%idx: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>, %cond: i1) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %off = scf.if %cond -> (tensor<32xi32, #b>) {
      scf.yield %range : tensor<32xi32, #b>
    } else {
      %tok = tt.load %idx : tensor<32x!tt.ptr<i32>, #b>
      scf.yield %tok : tensor<32xi32, #b>
    }
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p = tt.addptr %st, %off : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
    tt.return
  })"));
}

// The loaded tensor reaches the offset through several elementwise ops.
TEST_F(AddressDependsOnLoadedTensorTest, TransitiveThroughOps) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%idx: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<32xi32, #b>
    %c4 = arith.constant dense<4> : tensor<32xi32, #b>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %tok = tt.load %idx : tensor<32x!tt.ptr<i32>, #b>
    %blk = arith.divsi %tok, %c64 : tensor<32xi32, #b>
    %row = arith.muli %blk, %c4 : tensor<32xi32, #b>
    %off = arith.addi %row, %range : tensor<32xi32, #b>
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p = tt.addptr %st, %off : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
    tt.return
  })"));
}

// A non-load memory reader (atomic_rmw returns the old values) also counts.
TEST_F(AddressDependsOnLoadedTensorTest, NonLoadReader) {
  EXPECT_TRUE(probe(R"(
  tt.func @f(%ctr: tensor<32x!tt.ptr<i32>, #b>, %table: !tt.ptr<i32>, %mask: tensor<32xi1, #b>) {
    %one = arith.constant dense<1> : tensor<32xi32, #b>
    %old = tt.atomic_rmw add, acq_rel, gpu, %ctr, %one, %mask : (tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>, tensor<32xi1, #b>) -> tensor<32xi32, #b>
    %st = tt.splat %table : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p = tt.addptr %st, %old : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
    tt.return
  })"));
}

// A tensor read reduced to a scalar is uniform across the tile: the address
// stays structured, so this is not a gather.
TEST_F(AddressDependsOnLoadedTensorTest, TensorReducedToScalarIsUniform) {
  EXPECT_FALSE(probe(R"(
  tt.func @f(%lens: tensor<32x!tt.ptr<i32>, #b>, %in: !tt.ptr<i32>) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %l = tt.load %lens : tensor<32x!tt.ptr<i32>, #b>
    %sum = "tt.reduce"(%l) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %c: i32):
      %s = arith.addi %a, %c : i32
      tt.reduce.return %s : i32
    }) : (tensor<32xi32, #b>) -> i32
    %base = tt.addptr %in, %sum : !tt.ptr<i32>, i32
    %sb = tt.splat %base : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p = tt.addptr %sb, %range : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %v = tt.load %p {probe} : tensor<32x!tt.ptr<i32>, #b>
    tt.return
  })"));
}

// Structured loop-carried pointer: init and yield are both structured.
TEST_F(AddressDependsOnLoadedTensorTest, StructuredLoopIsNotGather) {
  EXPECT_FALSE(probe(R"(
  tt.func @f(%in: !tt.ptr<i32>, %lb: i32, %ub: i32, %cond: i1) {
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant dense<32> : tensor<32xi32, #b>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #b>
    %s = tt.splat %in : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #b>
    %p0 = tt.addptr %s, %range : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
    %r = scf.for %i = %lb to %ub step %c1 iter_args(%p = %p0) -> (tensor<32x!tt.ptr<i32>, #b>) : i32 {
      %q = scf.if %cond -> (tensor<32x!tt.ptr<i32>, #b>) {
        scf.yield %p : tensor<32x!tt.ptr<i32>, #b>
      } else {
        %pa = tt.addptr %p, %c32 : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
        scf.yield %pa : tensor<32x!tt.ptr<i32>, #b>
      }
      %v = tt.load %q {probe} : tensor<32x!tt.ptr<i32>, #b>
      %pn = tt.addptr %p, %c32 : tensor<32x!tt.ptr<i32>, #b>, tensor<32xi32, #b>
      scf.yield %pn : tensor<32x!tt.ptr<i32>, #b>
    }
    tt.return
  })"));
}

// Phase 1.6 Tests — knownReuse

TEST_F(ReuseAnalysisTest, Known_TrueFromSpatial) {
  // Mirror UnionTrueFromSpatial (line 116): DPAS dot-operand A with warp-
  // invariant K (dim 1), load outside any loop. Spatial known reuse fires,
  // temporal does not. Expected: knownReuse == true.
  OwningOpRef<ModuleOp> module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  auto dpasEnc = makeDpas(/*repeatCount=*/8, /*systolicDepth=*/8,
                          /*executionSize=*/16, /*opsPerChannel=*/2,
                          /*warpsPerCTA=*/{4, 1}, /*repCluster=*/{1, 1},
                          /*threadsPerWarp=*/16);
  auto dotOpEnc = ttg::DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpasEnc,
                                                   /*kWidth=*/1);
  auto resultTy = RankedTensorType::get({8, 16}, f16Ty, dotOpEnc);
  auto ptrTensorTy = RankedTensorType::get({8, 16}, ptrType, dotOpEnc);

  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);
  auto loadOp = makeLoad(splatPtr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_TRUE(analysis.knownReuse(loadOp));
  // Regression guard: anyReuse still true.
  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

TEST_F(ReuseAnalysisTest, Known_TrueFromTemporal) {
  // Mirror UnionTrueFromTemporal (line 149): 1-D blocked encoding with warps
  // tiling the only axis (no spatial reuse), but pointer is loop-invariant
  // inside scf.for (temporal proven reuse). Expected: knownReuse == true.
  OwningOpRef<ModuleOp> module = createModule();
  auto f16Ty = builder->getF16Type();
  auto ptrType = getPtrType(f16Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  auto blocked = makeBlocked(/*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
                             /*warpsPerCTA=*/{4}, /*order=*/{0});
  auto resultTy = RankedTensorType::get({128}, f16Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({128}, ptrType, blocked);

  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);

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

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  EXPECT_TRUE(analysis.knownReuse(loadOp));
  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

TEST_F(ReuseAnalysisTest, Known_FalseFromConservativeFallback_NonPow2) {
  // Central regression guard for the API split: encoded load with
  // non-power-of-2 shape (24x32). Spatial analysis falls back to fullAxisSet
  // (anyReuse = true), but knownCrossSubgroupReuse returns false. Expected:
  //   - getSpatial().getWarpInvariantOutDims(ty) returns {0, 1} (fallback)
  //   - getSpatial().knownCrossSubgroupReuse(ty) == false
  //   - knownReuse(loadOp) == false
  //   - anyReuse(loadOp) == true (conservative positive preserved)
  OwningOpRef<ModuleOp> module = createModule();
  auto f32Ty = builder->getF32Type();
  auto ptrType = getPtrType(f32Ty);
  auto funcOp = createFunctionWithReturn(*module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  // Blocked encoding with warpsPerCTA=[4,1]. If shape were 32x32, dim 1 would
  // be warp-invariant (known reuse). With actual shape 24x32, non-pow2 triggers
  // fallback.
  auto blocked =
      makeBlocked(/*sizePerThread=*/{1, 1}, /*threadsPerWarp=*/{1, 32},
                  /*warpsPerCTA=*/{4, 1}, /*order=*/{1, 0});
  auto resultTy = RankedTensorType::get({24, 32}, f32Ty, blocked);
  auto ptrTensorTy = RankedTensorType::get({24, 32}, ptrType, blocked);

  auto splatPtr = tt::SplatOp::create(*builder, loc, ptrTensorTy, basePtr);
  auto loadOp = makeLoad(splatPtr, resultTy);

  tti::ModuleAxisInfoAnalysis axisInfo(*module);
  tti::ModuleStrideAnalysis strideAnalysis(*module, axisInfo);
  ttgi::ReuseAnalysis analysis(*module, strideAnalysis);

  // Existing accessor returns fallback full axis set.
  EXPECT_THAT(analysis.getSpatial().getWarpInvariantOutDims(resultTy),
              ::testing::ElementsAre(0u, 1u));
  // New known accessor returns false.
  EXPECT_FALSE(analysis.getSpatial().knownCrossSubgroupReuse(resultTy));
  EXPECT_FALSE(analysis.knownReuse(loadOp));
  // Conservative anyReuse still true (regression guard).
  EXPECT_TRUE(analysis.anyReuse(loadOp));
}

} // namespace
