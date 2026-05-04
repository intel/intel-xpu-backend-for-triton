#include "intel/include/Analysis/AliasReuseAnalysis.h"
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

class AliasReuseAnalysisTest : public ::testing::Test {
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

TEST_F(AliasReuseAnalysisTest, TwoLoadsSameArgDifferentOffsets) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto offset1 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(0));
  auto ptr1 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset1);
  auto load1 =
      builder->create<triton::LoadOp>(loc, ptr1, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  auto offset2 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(16));
  auto ptr2 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset2);
  auto load2 =
      builder->create<triton::LoadOp>(loc, ptr2, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load1),
              ::testing::Contains(load2.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(load2),
              ::testing::Contains(load1.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, TwoLoadsDistinctArgs) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunction(module, "test_func", {ptrType, ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);
  auto arg1 = funcOp.getArgument(1);

  auto load1 =
      builder->create<triton::LoadOp>(loc, arg0, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);
  auto load2 =
      builder->create<triton::LoadOp>(loc, arg1, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load1), ::testing::IsEmpty());
  EXPECT_THAT(analysis.getAliasingMemOps(load2), ::testing::IsEmpty());
}

TEST_F(AliasReuseAnalysisTest, LoadAndStoreSameArg) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto i32Type = builder->getI32Type();
  auto offset1 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(0));
  auto ptr1 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset1);
  auto load =
      builder->create<triton::LoadOp>(loc, ptr1, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  auto offset2 = builder->create<arith::ConstantOp>(
      loc, i32Type, builder->getI32IntegerAttr(32));
  auto ptr2 = builder->create<triton::AddPtrOp>(loc, ptrType, arg0, offset2);
  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  auto storeOp = builder->create<triton::StoreOp>(
      loc, ptr2, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, LoadAndStoreDistinctArgs) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunction(module, "test_func", {ptrType, ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  auto loc = builder->getUnknownLoc();
  auto argA = funcOp.getArgument(0);
  auto argB = funcOp.getArgument(1);

  auto load =
      builder->create<triton::LoadOp>(loc, argA, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  builder->create<triton::StoreOp>(loc, argB, value,
                                   triton::CacheModifier::NONE,
                                   triton::EvictionPolicy::NORMAL);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load), ::testing::IsEmpty());
}

TEST_F(AliasReuseAnalysisTest, SCFForIterCarriedPointer_JoinsWithInit) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF16Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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
    builder->create<triton::LoadOp>(loc, iterPtr, triton::CacheModifier::NONE,
                                    triton::EvictionPolicy::NORMAL, false);
    builder->create<scf::YieldOp>(loc, ValueRange{iterPtr});
  }

  builder->setInsertionPointAfter(forOp);
  auto value = builder->create<arith::ConstantOp>(
      loc, builder->getF16Type(), builder->getF16FloatAttr(1.0));
  auto storeOp = builder->create<triton::StoreOp>(
      loc, argA, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

  builder->create<triton::ReturnOp>(loc);

  triton::LoadOp loadInLoop;
  forOp.getBody()->walk([&](triton::LoadOp op) { loadInLoop = op; });

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadInLoop),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, OpaqueLoadAliasesOpaqueAtomic) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getI32Type());
  auto funcOp = createFunction(module, "test_func", {ptrType, ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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
  auto loadOp = builder->create<triton::LoadOp>(
      loc, opaquePtr2.getResult(), triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadOp),
              ::testing::Contains(atomicOp.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, AtomicResolvedDistinctFromLoad) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getI32Type());
  auto funcOp = createFunction(module, "test_func", {ptrType, ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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
  auto load =
      builder->create<triton::LoadOp>(loc, argB, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(load), ::testing::IsEmpty());
}

TEST_F(AliasReuseAnalysisTest, ConvertLayoutPointerPassThrough) {
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

  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadedValue),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, ThreeLoadsSameArgReturnsBoth) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  auto loc = builder->getUnknownLoc();
  auto arg0 = funcOp.getArgument(0);

  auto loadA =
      builder->create<triton::LoadOp>(loc, arg0, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);
  auto loadB =
      builder->create<triton::LoadOp>(loc, arg0, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);
  auto loadC =
      builder->create<triton::LoadOp>(loc, arg0, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Contains(loadB.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadA),
              ::testing::Contains(loadC.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Contains(loadA.getOperation()));
  EXPECT_THAT(analysis.getAliasingMemOps(loadB),
              ::testing::Contains(loadC.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, DescriptorLoadAndDescriptorStoreSameBase) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(storeOp.getOperation()));
}

TEST_F(AliasReuseAnalysisTest, DescriptorLoadAndRawLoadSameBase) {
  auto module = createModule();
  auto ptrType = getPtrType(builder->getF32Type());
  auto funcOp = createFunction(module, "test_func", {ptrType});

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

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
  auto rawload =
      builder->create<triton::LoadOp>(loc, base, triton::CacheModifier::NONE,
                                      triton::EvictionPolicy::NORMAL, false);

  builder->create<triton::ReturnOp>(loc, ValueRange{});

  mlir::triton::intel::AliasReuseAnalysis analysis(funcOp);
  EXPECT_THAT(analysis.getAliasingMemOps(dload),
              ::testing::Contains(rawload.getOperation()));
}

} // namespace
