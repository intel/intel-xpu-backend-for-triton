#include "intel/include/Analysis/AliasAnalysis.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;

namespace {

class AliasAnalysisTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<triton::TritonDialect>();
    ctx.getOrLoadDialect<triton::gpu::TritonGPUDialect>();
    ctx.getOrLoadDialect<triton::gpu::intel::TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  ModuleOp createModule() {
    auto loc = builder->getUnknownLoc();
    return ModuleOp::create(loc);
  }

  triton::FuncOp createFunction(ModuleOp module, StringRef name,
                                ArrayRef<Type> argTypes) {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToEnd(module.getBody());

    auto funcType = builder->getFunctionType(argTypes, {});
    auto funcOp = builder->create<triton::FuncOp>(builder->getUnknownLoc(),
                                                  name, funcType);
    funcOp.addEntryBlock();
    builder->setInsertionPointToStart(&funcOp.getBody().front());

    return funcOp;
  }

  /// Like createFunction, but appends an empty `tt.return` at the end of
  /// the entry block and leaves the insertion point immediately BEFORE
  /// the return so the body can be built without caring about the
  /// terminator.
  triton::FuncOp createFunctionWithReturn(ModuleOp module, StringRef name,
                                          ArrayRef<Type> argTypes) {
    auto funcOp = createFunction(module, name, argTypes);
    auto loc = builder->getUnknownLoc();
    auto ret = builder->create<triton::ReturnOp>(loc, ValueRange{});
    builder->setInsertionPoint(ret);
    return funcOp;
  }

  /// Creates a tt.load with default cache/eviction attributes.
  triton::LoadOp makeLoad(Value ptr) {
    return builder->create<triton::LoadOp>(
        builder->getUnknownLoc(), ptr, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, /*isVolatile=*/false);
  }

  Type getPtrType(Type elemType) {
    return triton::PointerType::get(elemType, 1);
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// ===----------------------------------------------------------------------===//
// Test Alias Detection
// ===----------------------------------------------------------------------===//

TEST_F(AliasAnalysisTest, TwoLoadsSameArgDifferentOffsets) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto offset1 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(0));
  auto ptr1 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset1);
  auto load1 = makeLoad(ptr1);

  auto offset2 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(16));
  auto ptr2 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset2);
  auto load2 = makeLoad(ptr2);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load1),
              ::testing::Contains(load2.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(load2),
              ::testing::Contains(load1.getOperation()));
}

TEST_F(AliasAnalysisTest, TwoLoadsDistinctArgs) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);
  auto arg1 = funcOp.getArgument(1);

  auto load1 = makeLoad(arg0);
  auto load2 = makeLoad(arg1);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load1), ::testing::IsEmpty());
  EXPECT_THAT(analysis.getAliasingMemOps(load2), ::testing::IsEmpty());
}

TEST_F(AliasAnalysisTest, LoadAndStoreSameArg) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto offset1 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(0));
  auto ptr1 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset1);
  auto load = makeLoad(ptr1);

  auto offset2 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(32));
  auto ptr2 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset2);
  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  auto storeOp = builder->create<triton::StoreOp>(
      loc, ptr2, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasAnalysisTest, LoadAndStoreDistinctArgs) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);
  auto argB = funcOp.getArgument(1);

  auto load = makeLoad(argA);

  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  builder->create<triton::StoreOp>(loc, argB, value,
                                   triton::CacheModifier::NONE,
                                   triton::EvictionPolicy::NORMAL);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load), ::testing::IsEmpty());
}

TEST_F(AliasAnalysisTest, SCFForIterCarriedPointer_JoinsWithInit) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto lb = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(0));
  auto ub = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(10));
  auto step = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(1));

  auto forOp = builder->create<scf::ForOp>(loc, lb, ub, step, ValueRange{argA});

  {
    OpBuilder::InsertionGuard loopGuard(*builder);
    builder->setInsertionPointToStart(forOp.getBody());
    auto iterPtr = forOp.getRegionIterArg(0);
    makeLoad(iterPtr);
    builder->create<scf::YieldOp>(loc, ValueRange{iterPtr});
  }

  builder->setInsertionPoint(funcOp.front().getTerminator());
  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  auto storeOp = builder->create<triton::StoreOp>(
      loc, argA, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

  triton::LoadOp loadInLoop;
  forOp.getBody()->walk([&](triton::LoadOp op) { loadInLoop = op; });

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadInLoop),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasAnalysisTest, OpaqueLoadAliasesOpaqueAtomic) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getI32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);
  auto arg1 = funcOp.getArgument(1);

  // Create an opaque pointer via arith.select (not recognized by analysis)
  auto i1Type = builder->getI1Type();
  auto condition = builder->create<arith::ConstantOp>(
      loc, i1Type, builder->getIntegerAttr(i1Type, 1));
  auto opaquePtr = builder->create<arith::SelectOp>(loc, condition, arg0, arg1);

  // Atomic on opaque pointer
  auto atomicValue = builder->create<arith::ConstantOp>(
      loc, builder->getI32Type(), builder->getI32IntegerAttr(1));
  auto atomicOp = builder->create<triton::AtomicRMWOp>(
      loc, builder->getI32Type(), triton::RMWOp::ADD, opaquePtr.getResult(),
      atomicValue, /*mask=*/Value(), /*sem=*/triton::MemSemantic::RELAXED,
      /*scope=*/triton::MemSyncScope::GPU);

  // Load from another opaque pointer (also via select)
  auto arg0_2 = funcOp.getArgument(0);
  auto arg1_2 = funcOp.getArgument(1);
  auto opaquePtr2 =
      builder->create<arith::SelectOp>(loc, condition, arg1_2, arg0_2);
  auto loadOp = makeLoad(opaquePtr2.getResult());

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadOp),
              ::testing::Contains(atomicOp.getOperation()));
}

TEST_F(AliasAnalysisTest, AtomicResolvedDistinctFromLoad) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getI32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);
  auto argB = funcOp.getArgument(1);

  // Atomic on resolved pointer A
  auto atomicValue = builder->create<arith::ConstantOp>(
      loc, builder->getI32Type(), builder->getI32IntegerAttr(1));
  builder->create<triton::AtomicRMWOp>(loc, builder->getI32Type(),
                                       triton::RMWOp::ADD, argA, atomicValue,
                                       /*mask=*/Value(),
                                       /*sem=*/triton::MemSemantic::RELAXED,
                                       /*scope=*/triton::MemSyncScope::GPU);

  // Load from resolved pointer B (distinct)
  auto load = makeLoad(argB);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load), ::testing::IsEmpty());
}

TEST_F(AliasAnalysisTest, OpaquePointerAliasesResolvedPointer) {
  // An opaque pointer (produced by an op the analysis doesn't model, e.g.
  // arith.select) has an unresolved origin. It must conservatively MayAlias
  // every tracked pointer — including resolved pointers derived directly
  // from a function argument that the opaque pointer could equal at runtime.
  auto module = createModule();
  auto ptrType = getPtrType(builder->getI32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);
  auto arg1 = funcOp.getArgument(1);

  // Opaque pointer: could be %arg0 or %arg1 at runtime.
  auto i1Type = builder->getI1Type();
  auto condition = builder->create<arith::ConstantOp>(
      loc, i1Type, builder->getIntegerAttr(i1Type, 1));
  auto opaquePtr = builder->create<arith::SelectOp>(loc, condition, arg0, arg1);

  // Load from the opaque pointer.
  auto opaqueLoad = makeLoad(opaquePtr.getResult());

  // Load from resolved %arg0 (which the opaque pointer may equal).
  auto resolvedLoadA = makeLoad(arg0);
  // Load from resolved %arg1 (same reasoning).
  auto resolvedLoadB = makeLoad(arg1);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  // Opaque pointer must MayAlias both resolved pointers.
  EXPECT_THAT(analysis.getAliasingMemOps(opaqueLoad),
              ::testing::Contains(resolvedLoadA.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(opaqueLoad),
              ::testing::Contains(resolvedLoadB.getOperation()));
  // And symmetrically: each resolved load must see the opaque load as a peer.
  EXPECT_THAT(analysis.getAliasingMemOps(resolvedLoadA),
              ::testing::Contains(opaqueLoad.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(resolvedLoadB),
              ::testing::Contains(opaqueLoad.getOperation()));
  // Resolved-vs-resolved distinct args still NoAlias.
  EXPECT_THAT(
      analysis.getAliasingMemOps(resolvedLoadA),
      ::testing::Not(::testing::Contains(resolvedLoadB.getOperation())));
}

TEST_F(AliasAnalysisTest, ConvertLayoutPointerPassThrough) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());

  // Create a tensor-of-pointer type for load operand
  auto tensorPtrType = RankedTensorType::get({128}, ptrType);

  // Need a minimal blocked encoding
  auto cgaLayout = triton::gpu::CGAEncodingAttr::fromSplitParams(
      &ctx, /*CTAsPerCGA=*/{1}, /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto blockedEnc = triton::gpu::BlockedEncodingAttr::get(
      &ctx, /*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
      /*warpsPerCTA=*/{1}, /*order=*/{0}, cgaLayout);
  auto tensorPtrTypeWithEnc = RankedTensorType::get({128}, ptrType, blockedEnc);

  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto basePtr = funcOp.getArgument(0);

  // Create splat (tensor-of-pointer from scalar pointer)
  auto splatPtr = builder->create<triton::SplatOp>(loc, tensorPtrType, basePtr);

  // Pass through convert_layout (on the pointer tensor)
  auto convertedPtr = builder->create<triton::gpu::ConvertLayoutOp>(
      loc, tensorPtrTypeWithEnc, splatPtr);

  // Load using the converted pointer
  auto loadedValue = builder->create<triton::LoadOp>(
      loc, convertedPtr, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);

  // Store using the base pointer (through a different chain)
  auto i32Type = builder->getI32Type();
  auto offset = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(64));
  auto storePtr =
      builder->create<triton::AddPtrOp>(loc, ptrType, basePtr, offset);
  auto storeValue = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  auto storeOp = builder->create<triton::StoreOp>(
      loc, storePtr, storeValue, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadedValue),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasAnalysisTest, ThreeLoadsSameArgReturnsBoth) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto loadA = makeLoad(arg0);
  auto loadB = makeLoad(arg0);
  auto loadC = makeLoad(arg0);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Contains(loadB.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Contains(loadC.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Contains(loadA.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Contains(loadC.getOperation()));
}

TEST_F(AliasAnalysisTest, DescriptorLoadAndDescriptorStoreSameBase) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto base = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto i64Type = builder->getI64Type();
  auto c128 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(128));
  auto c64 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                builder->getI32IntegerAttr(64));
  auto c64_i64 = builder->create<arith::ConstantOp>(
      loc, i64Type, builder->getI64IntegerAttr(64));
  auto c1 = builder->create<arith::ConstantOp>(loc, i64Type,
                                               builder->getI64IntegerAttr(1));
  auto idx0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto idx1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));

  auto tensorType = RankedTensorType::get({128, 64}, builder->getF32Type());
  auto descType = triton::TensorDescType::get({128, 64}, builder->getF32Type(),
                                              Attribute{});
  auto desc = builder->create<triton::MakeTensorDescOp>(
      loc, descType, base, ValueRange{c128, c64}, ValueRange{c64_i64, c1});
  auto dload = builder->create<triton::DescriptorLoadOp>(
      loc, tensorType, desc, ValueRange{idx0, idx1});
  auto val = builder->create<arith::ConstantOp>(
      loc, tensorType,
      DenseElementsAttr::get(tensorType, builder->getF32FloatAttr(1.0)));
  auto storeOp = builder->create<triton::DescriptorStoreOp>(
      loc, desc, val, ValueRange{idx0, idx1});

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasAnalysisTest, DescriptorLoadAndRawLoadSameBase) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto base = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto i64Type = builder->getI64Type();
  auto c128 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(128));
  auto c64 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                builder->getI32IntegerAttr(64));
  auto c64_i64 = builder->create<arith::ConstantOp>(
      loc, i64Type, builder->getI64IntegerAttr(64));
  auto c1 = builder->create<arith::ConstantOp>(loc, i64Type,
                                               builder->getI64IntegerAttr(1));
  auto idx0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto idx1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));

  auto tensorType = RankedTensorType::get({128, 64}, builder->getF32Type());
  auto descType = triton::TensorDescType::get({128, 64}, builder->getF32Type(),
                                              Attribute{});
  auto desc = builder->create<triton::MakeTensorDescOp>(
      loc, descType, base, ValueRange{c128, c64}, ValueRange{c64_i64, c1});
  auto dload = builder->create<triton::DescriptorLoadOp>(
      loc, tensorType, desc, ValueRange{idx0, idx1});
  auto rawload = makeLoad(base);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(rawload.getOperation()));
}

TEST_F(AliasAnalysisTest, DescriptorLoadThroughSCFForIterArg) {
  // When a descriptor flows through scf.for iter_args, the
  // tt.descriptor_load's getDesc() is a block argument, not a direct
  // MakeTensorDescOp result. findDefiningOpOfType<> must trace through the
  // iter_arg back to the original descriptor so the op is not dropped and
  // its base pointer is resolved correctly.
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunctionWithReturn(module, "test_func", {ptrType});
  auto loc = builder->getUnknownLoc();
  auto base = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto i64Type = builder->getI64Type();
  auto c128 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(128));
  auto c64 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                builder->getI32IntegerAttr(64));
  auto c64_i64 = builder->create<arith::ConstantOp>(
      loc, i64Type, builder->getI64IntegerAttr(64));
  auto c1 = builder->create<arith::ConstantOp>(loc, i64Type,
                                               builder->getI64IntegerAttr(1));
  auto idx0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto idx1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));

  auto tensorType = RankedTensorType::get({128, 64}, builder->getF32Type());
  auto descType = triton::TensorDescType::get({128, 64}, builder->getF32Type(),
                                              Attribute{});
  auto desc = builder->create<triton::MakeTensorDescOp>(
      loc, descType, base, ValueRange{c128, c64}, ValueRange{c64_i64, c1});

  auto lb = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(0));
  auto ub = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(10));
  auto step = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(1));
  auto forOp = builder->create<scf::ForOp>(loc, lb, ub, step, ValueRange{desc});

  triton::DescriptorLoadOp dload;
  {
    OpBuilder::InsertionGuard loopGuard(*builder);
    builder->setInsertionPointToStart(forOp.getBody());
    auto iterDesc = forOp.getRegionIterArg(0);
    dload = builder->create<triton::DescriptorLoadOp>(loc, tensorType, iterDesc,
                                                      ValueRange{idx0, idx1});
    builder->create<scf::YieldOp>(loc, ValueRange{iterDesc});
  }

  builder->setInsertionPoint(funcOp.front().getTerminator());
  auto rawload = makeLoad(base);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  // The descriptor_load inside the loop must resolve through the iter_arg
  // to `base` and alias the raw load from `base`.
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(rawload.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(rawload),
              ::testing::Contains(dload.getOperation()));
}

// ===----------------------------------------------------------------------===//
// New Tests — Verifying Extended Analysis Behavior
// ===----------------------------------------------------------------------===//

TEST_F(AliasAnalysisTest, DescriptorThroughSCFIfMismatch) {
  // When a descriptor escapes an scf.if with mismatched branches (two
  // MakeTensorDescOps from different base pointers), the descriptor
  // becomes opaque and should conservatively MayAlias loads from either
  // base.
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp =
      createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto baseA = funcOp.getArgument(0);
  auto baseB = funcOp.getArgument(1);

  auto i1Type = builder->getI1Type();
  auto i32Type = builder->getI32Type();
  auto i64Type = builder->getI64Type();
  auto condition = builder->create<arith::ConstantOp>(
      loc, i1Type, builder->getIntegerAttr(i1Type, 1));

  auto c128 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(128));
  auto c64 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                builder->getI32IntegerAttr(64));
  auto c64_i64 = builder->create<arith::ConstantOp>(
      loc, i64Type, builder->getI64IntegerAttr(64));
  auto c1 = builder->create<arith::ConstantOp>(loc, i64Type,
                                               builder->getI64IntegerAttr(1));

  auto tensorType = RankedTensorType::get({128, 64}, builder->getF32Type());
  auto descType = triton::TensorDescType::get({128, 64}, builder->getF32Type(),
                                              Attribute{});

  // scf.if with mismatched descriptor branches
  auto ifOp = builder->create<scf::IfOp>(loc, descType, condition, true);
  {
    OpBuilder::InsertionGuard thenGuard(*builder);
    builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto descA = builder->create<triton::MakeTensorDescOp>(
        loc, descType, baseA, ValueRange{c128, c64}, ValueRange{c64_i64, c1});
    builder->create<scf::YieldOp>(loc, ValueRange{descA.getResult()});
  }
  {
    OpBuilder::InsertionGuard elseGuard(*builder);
    builder->setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto descB = builder->create<triton::MakeTensorDescOp>(
        loc, descType, baseB, ValueRange{c128, c64}, ValueRange{c64_i64, c1});
    builder->create<scf::YieldOp>(loc, ValueRange{descB.getResult()});
  }

  auto opaqueDesc = ifOp.getResult(0);
  auto idx0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto idx1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto dload = builder->create<triton::DescriptorLoadOp>(
      loc, tensorType, opaqueDesc, ValueRange{idx0, idx1});

  auto loadA = makeLoad(baseA);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  // The opaque descriptor should MayAlias the raw load from baseA.
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(loadA.getOperation()));
}

TEST_F(AliasAnalysisTest, UnmodeledMemoryOpIsPeer) {
  // Verify that ops implementing MemoryEffectOpInterface with Read/Write
  // effects (but not in the 9 modeled types) are tracked as universal peers.
  // `tt.print` has MemWrite<GlobalMemory> and takes no pointer operands
  // (only a string prefix + optional variadic scalar args), so it becomes
  // a tracked op with a null pointer slot — a universal MayAlias peer.
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp =
      createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);
  auto argB = funcOp.getArgument(1);

  auto loadA = makeLoad(argA);
  auto printOp = builder->create<triton::PrintOp>(
      loc, builder->getStringAttr("dbg"), /*hex=*/builder->getBoolAttr(false),
      ValueRange{}, builder->getDenseI32ArrayAttr({}));
  auto loadB = makeLoad(argB);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  // Distinct-arg loads still NoAlias each other.
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Not(::testing::Contains(loadB.getOperation())));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Not(::testing::Contains(loadA.getOperation())));
  // The interface-tracked print op is a universal peer of both loads.
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Contains(printOp.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Contains(printOp.getOperation()));
  // And symmetrically: the print op reports both loads as peers.
  EXPECT_THAT(analysis.getAliasingMemOps(printOp.getOperation()),
              ::testing::Contains(loadA.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(printOp.getOperation()),
              ::testing::Contains(loadB.getOperation()));
}

TEST_F(AliasAnalysisTest, NonMemoryOpsDoNotPessimize) {
  // Verify that a function containing only pure ops (no memory effects)
  // does not pessimize aliasing. Two distinct-arg loads should remain
  // NoAlias (peer sets empty).
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp =
      createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);
  auto argB = funcOp.getArgument(1);

  auto load1 = makeLoad(argA);

  // Pure ops
  auto i32Type = builder->getI32Type();
  auto c0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(0));
  auto c1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                               builder->getI32IntegerAttr(1));
  auto sum = builder->create<arith::AddIOp>(loc, c0, c1);
  (void)sum;

  auto load2 = makeLoad(argB);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load1), ::testing::IsEmpty());
  EXPECT_THAT(analysis.getAliasingMemOps(load2), ::testing::IsEmpty());
}

TEST_F(AliasAnalysisTest, OpaqueDescriptorPropagatesUnknown) {
  // Regression test for isPointerLike(TensorDescType) — a descriptor that
  // escapes control flow (via arith.select) should propagate Unknown and
  // conservatively MayAlias loads from either candidate base.
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp =
      createFunctionWithReturn(module, "test_func", {ptrType, ptrType});
  auto loc = builder->getUnknownLoc();
  auto baseA = funcOp.getArgument(0);
  auto baseB = funcOp.getArgument(1);

  auto i1Type = builder->getI1Type();
  auto i32Type = builder->getI32Type();
  auto i64Type = builder->getI64Type();
  auto condition = builder->create<arith::ConstantOp>(
      loc, i1Type, builder->getIntegerAttr(i1Type, 1));

  auto c128 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(128));
  auto c64 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                builder->getI32IntegerAttr(64));
  auto c64_i64 = builder->create<arith::ConstantOp>(
      loc, i64Type, builder->getI64IntegerAttr(64));
  auto c1 = builder->create<arith::ConstantOp>(loc, i64Type,
                                               builder->getI64IntegerAttr(1));

  auto tensorType = RankedTensorType::get({128, 64}, builder->getF32Type());
  auto descType = triton::TensorDescType::get({128, 64}, builder->getF32Type(),
                                              Attribute{});

  auto descA = builder->create<triton::MakeTensorDescOp>(
      loc, descType, baseA, ValueRange{c128, c64}, ValueRange{c64_i64, c1});
  auto descB = builder->create<triton::MakeTensorDescOp>(
      loc, descType, baseB, ValueRange{c128, c64}, ValueRange{c64_i64, c1});

  // Opaque descriptor via arith.select (not recognized by findDefiningOpOfType)
  auto opaqueDesc =
      builder->create<arith::SelectOp>(loc, condition, descA, descB);

  auto idx0 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto idx1 = builder->create<arith::ConstantOp>(loc, i32Type,
                                                 builder->getI32IntegerAttr(0));
  auto dload = builder->create<triton::DescriptorLoadOp>(
      loc, tensorType, opaqueDesc.getResult(), ValueRange{idx0, idx1});

  auto loadA = makeLoad(baseA);

  mlir::triton::intel::AliasAnalysis analysis(funcOp);
  // The opaque descriptor should MayAlias loadA (and symmetrically loadB).
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(loadA.getOperation()));
}

} // namespace
