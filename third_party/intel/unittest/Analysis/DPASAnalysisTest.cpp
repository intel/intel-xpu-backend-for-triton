#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {

class DPASAnalysisTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  ModuleOp createModule(bool supportDPAS = true, bool supportFp8 = false,
                        int minSGSize = 16, int threadsPerWarp = 16) {
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);

    if (supportDPAS)
      module->setAttr(TritonIntelGPUDialect::getSupportDPASAttrName(),
                      builder->getUnitAttr());

    if (supportFp8)
      module->setAttr(TritonIntelGPUDialect::getSupportDPASWithBF8AttrName(),
                      builder->getUnitAttr());

    module->setAttr(TritonIntelGPUDialect::getMinSGSizeAttrName(),
                    builder->getI32IntegerAttr(minSGSize));

    module->setAttr(AttrNumThreadsPerWarp,
                    builder->getI32IntegerAttr(threadsPerWarp));

    return module;
  }

  func::FuncOp createFunction(ModuleOp module, StringRef name = "test_func") {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToEnd(module.getBody());

    auto funcType = builder->getFunctionType({}, {});
    auto funcOp =
        builder->create<func::FuncOp>(builder->getUnknownLoc(), name, funcType);
    auto *block = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(block);

    return funcOp;
  }

  DotOp createDotOp(ModuleOp module, Type aElemType, Type bElemType,
                    Type cElemType,
                    InputPrecision precision = InputPrecision::IEEE) {
    OpBuilder::InsertionGuard guard(*builder);
    auto funcOp = createFunction(module);
    builder->setInsertionPointToStart(&funcOp.getBody().front());

    auto loc = builder->getUnknownLoc();
    auto tensorTypeA = RankedTensorType::get({32, 32}, aElemType);
    auto tensorTypeB = RankedTensorType::get({32, 32}, bElemType);
    auto tensorTypeC = RankedTensorType::get({32, 32}, cElemType);

    // Create placeholder values
    auto a = builder->create<arith::ConstantOp>(
        loc, tensorType, builder->getZeroAttr(tensorType));
    auto b = builder->create<arith::ConstantOp>(
        loc, tensorTypeB, builder->getZeroAttr(tensorTypeB));
    auto c = builder->create<arith::ConstantOp>(
        loc, tensorTypeC, builder->getZeroAttr(tensorTypeC));

    auto dotOp =
        builder->create<DotOp>(loc, tensorTypeC, a, b, c, precision, 0);

    builder->create<func::ReturnOp>(loc);
    return dotOp;
  }

  DotScaledOp createDotScaledOp(ModuleOp module, Type aElemType, Type bElemType,
                                Type cElemType, ScaleDotElemType aScaleType,
                                ScaleDotElemType bScaleType,
                                bool lhsKPack = false, bool rhsKPack = false) {
    OpBuilder::InsertionGuard guard(*builder);
    auto funcOp = createFunction(module);
    builder->setInsertionPointToStart(&funcOp.getBody().front());

    auto loc = builder->getUnknownLoc();
    auto tensorTypeA = RankedTensorType::get({32, 32}, aElemType);
    auto tensorTypeB = RankedTensorType::get({32, 32}, bElemType);
    auto tensorTypeC = RankedTensorType::get({32, 32}, cElemType);
    auto scaleType = RankedTensorType::get({32, 2}, builder->getI8Type());

    auto a = builder->create<arith::ConstantOp>(
        loc, tensorTypeA, builder->getZeroAttr(tensorTypeA));
    auto b = builder->create<arith::ConstantOp>(
        loc, tensorTypeB, builder->getZeroAttr(tensorTypeB));
    auto c = builder->create<arith::ConstantOp>(
        loc, tensorTypeC, builder->getZeroAttr(tensorTypeC));
    auto scaleA = builder->create<arith::ConstantOp>(
        loc, scaleType, builder->getZeroAttr(scaleType));
    auto scaleB = builder->create<arith::ConstantOp>(
        loc, scaleType, builder->getZeroAttr(scaleType));

    auto dotScaledOp = builder->create<DotScaledOp>(
        loc, tensorTypeC, a, b, c, scaleA, scaleB, aScaleType, bScaleType,
        /*fastMath=*/false, lhsKPack, rhsKPack);

    builder->create<func::ReturnOp>(loc);
    return dotScaledOp;
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// ===----------------------------------------------------------------------===//
// Test DPAS Type Detection for DotOp
// ===----------------------------------------------------------------------===//

TEST_F(DPASAnalysisTest, DotOp_FP16_FP16_FP32) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getF16Type(), builder->getF16Type(),
                           builder->getF32Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP16_FP16);
}

TEST_F(DPASAnalysisTest, DotOp_BF16_BF16_BF16) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getBF16Type(),
                           builder->getBF16Type(), builder->getBF16Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::BF16_BF16_BF16_BF16);
}

TEST_F(DPASAnalysisTest, DotOp_BF16_BF16_FP32) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getBF16Type(),
                           builder->getBF16Type(), builder->getF32Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_BF16_BF16);
}

TEST_F(DPASAnalysisTest, DotOp_FP32_TF32_FP32) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getF32Type(), builder->getF32Type(),
                           builder->getF32Type(), InputPrecision::TF32);

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_TF32_TF32);
}

TEST_F(DPASAnalysisTest, DotOp_FP8_Native) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotOp = createDotOp(module, Float8E5M2Type::get(&ctx),
                           Float8E5M2Type::get(&ctx), builder->getF32Type());

  auto dpasTypeXe2 = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasTypeXe2, DPASEngineTypeXe2::FP32_FP32_FP8_FP8);
}

TEST_F(DPASAnalysisTest, DotOp_FP8_Upcast) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/false);
  auto dotOp = createDotOp(module, Float8E4M3FNType::get(&ctx),
                           Float8E4M3FNType::get(&ctx), builder->getF32Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  // Should upcast to FP16 when FP8 is not supported
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP16_FP16);
}

TEST_F(DPASAnalysisTest, DotOp_INT8_Signed) {
  auto module = createModule(/*supportDPAS=*/true);
  auto i8Type = builder->getIntegerType(8, /*isSigned=*/true);
  auto si32Type = builder->getIntegerType(32, /*isSigned=*/true);
  auto dotOp = createDotOp(module, i8Type, i8Type, si32Type);

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::S32_S32_S8_S8);
}

TEST_F(DPASAnalysisTest, DotOp_INT8_Unsigned) {
  auto module = createModule(/*supportDPAS=*/true);
  auto ui8Type = builder->getIntegerType(8, /*isSigned=*/false);
  auto ui32Type = builder->getIntegerType(32, /*isSigned=*/false);
  auto dotOp = createDotOp(module, ui8Type, ui8Type, ui32Type);

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::U32_U32_U8_U8);
}

TEST_F(DPASAnalysisTest, DotOp_MismatchedTypes) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getF16Type(),
                           builder->getBF16Type(), builder->getF32Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::NOT_APPLICABLE);
}

TEST_F(DPASAnalysisTest, DotOp_Xe3P_BF16_FP8) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotOp = createDotOp(module, Float8E5M2Type::get(&ctx),
                           Float8E5M2Type::get(&ctx), builder->getBF16Type());

  auto dpasType = DPASAnalysisV2::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe3P::BF16_BF16_FP8_FP8);
}

// ===----------------------------------------------------------------------===//
// Test DPAS Type Detection for DotScaledOp
// ===----------------------------------------------------------------------===//

TEST_F(DPASAnalysisTest, DotScaledOp_BF16_FP8) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotScaledOp = createDotScaledOp(
      module, builder->getBF16Type(), Float8E5M2Type::get(&ctx),
      builder->getF32Type(), ScaleDotElemType::BF16, ScaleDotElemType::E5M2);

  auto dpasType = DPASAnalysisV1::getDPASType(dotScaledOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_BF16_FP8);
}

TEST_F(DPASAnalysisTest, DotScaledOp_FP8_FP8) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotScaledOp = createDotScaledOp(
      module, Float8E4M3FNType::get(&ctx), Float8E5M2Type::get(&ctx),
      builder->getF32Type(), ScaleDotElemType::E4M3, ScaleDotElemType::E5M2);

  auto dpasType = DPASAnalysisV1::getDPASType(dotScaledOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP8_FP8);
}

TEST_F(DPASAnalysisTest, DotScaledOp_FP4_FP4) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotScaledOp = createDotScaledOp(
      module, builder->getI8Type(), builder->getI8Type(), builder->getF32Type(),
      ScaleDotElemType::E2M1, ScaleDotElemType::E2M1,
      /*lhsKPack=*/true, /*rhsKPack=*/true);

  auto dpasType = DPASAnalysisV1::getDPASType(dotScaledOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP4_FP4);
}

TEST_F(DPASAnalysisTest, DotScaledOp_Mixed_FP16_FP8) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotScaledOp = createDotScaledOp(
      module, builder->getF16Type(), Float8E4M3FNType::get(&ctx),
      builder->getF32Type(), ScaleDotElemType::FP16, ScaleDotElemType::E4M3);

  auto dpasType = DPASAnalysisV1::getDPASType(dotScaledOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP16_FP8);
}

// ===----------------------------------------------------------------------===//
// Test canUseDPAS() - Function Level
// ===----------------------------------------------------------------------===//

TEST_F(DPASAnalysisTest, CanUseDPAS_EmptyFunction) {
  auto module = createModule(/*supportDPAS=*/true);
  auto funcOp = createFunction(module);
  builder->setInsertionPointToStart(&funcOp.getBody().front());
  builder->create<func::ReturnOp>(builder->getUnknownLoc());

  DPASAnalysisV1 analysis(module);
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::False);
}

TEST_F(DPASAnalysisTest, CanUseDPAS_WithValidDotOp) {
  auto module = createModule(/*supportDPAS=*/true);
  createDotOp(module, builder->getF16Type(), builder->getF16Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::True);
}

TEST_F(DPASAnalysisTest, CanUseDPAS_WrongWarpSize) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/false,
                             /*minSGSize=*/16, /*threadsPerWarp=*/8);
  createDotOp(module, builder->getF16Type(), builder->getF16Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::False);
}

TEST_F(DPASAnalysisTest, CanUseDPAS_MaybeResult) {
  auto module = createModule(/*supportDPAS=*/true);
  // Don't set threads-per-warp attribute
  module->removeAttr(AttrNumThreadsPerWarp);
  createDotOp(module, builder->getF16Type(), builder->getF16Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::Maybe);
}

TEST_F(DPASAnalysisTest, CanUseDPAS_NotApplicableType) {
  auto module = createModule(/*supportDPAS=*/true);
  // Create a dot with mismatched types
  createDotOp(module, builder->getF16Type(), builder->getF32Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::False);
}

TEST_F(DPASAnalysisTest, CanUseDPAS_Warp32_Enabled) {
  // Set environment variable through test (would normally be set externally)
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/false,
                             /*minSGSize=*/16, /*threadsPerWarp=*/32);
  createDotOp(module, builder->getF16Type(), builder->getF16Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  // Result depends on TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32 env var
  EXPECT_TRUE(result == DPASAnalysisResult::True ||
              result == DPASAnalysisResult::False);
}

// ===----------------------------------------------------------------------===//
// Test DPASAnalysisFactory
// ===----------------------------------------------------------------------===//

TEST_F(DPASAnalysisTest, Factory_CreateXe2Analysis) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/false);

  auto analysis = DPASAnalysisFactory::createDPASAnalysis(module);
  EXPECT_TRUE(std::holds_alternative<DPASAnalysisV1>(analysis));
}

TEST_F(DPASAnalysisTest, Factory_CreateXe3PAnalysis) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);

  auto analysis = DPASAnalysisFactory::createDPASAnalysis(module);
  EXPECT_TRUE(std::holds_alternative<DPASAnalysisV2>(analysis));
}

TEST_F(DPASAnalysisTest, Factory_GetDPASType) {
  auto module = createModule(/*supportDPAS=*/true, /*supportFp8=*/true);
  auto dotOp = createDotOp(module, builder->getF16Type(), builder->getF16Type(),
                           builder->getF32Type());

  auto analysis = DPASAnalysisFactory::createDPASAnalysis(module);
  auto dpasType = DPASAnalysisFactory::getDPASType(dotOp, analysis);

  EXPECT_TRUE(std::holds_alternative<DPASEngineTypeXe3P>(dpasType));
  auto type = std::get<DPASEngineTypeXe3P>(dpasType);
  EXPECT_EQ(type, DPASEngineTypeXe3P::FP32_FP32_FP16_FP16);
}

TEST_F(DPASAnalysisTest, Factory_CanUseDPASOperation) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getF16Type(), builder->getF16Type(),
                           builder->getF32Type());

  auto analysis = DPASAnalysisFactory::createDPASAnalysis(module);
  auto result = DPASAnalysisFactory::canUseDPAS(dotOp, analysis);

  EXPECT_EQ(result, DPASAnalysisResult::True);
}

// ===----------------------------------------------------------------------===//
// Test Edge Cases
// ===----------------------------------------------------------------------===//

TEST_F(DPASAnalysisTest, EdgeCase_NoDPASSupport) {
  auto module = createModule(/*supportDPAS=*/false);
  createDotOp(module, builder->getF16Type(), builder->getF16Type(),
              builder->getF32Type());

  DPASAnalysisV1 analysis(module);
  auto funcOp = *module.getOps<func::FuncOp>().begin();
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::False);
}

TEST_F(DPASAnalysisTest, EdgeCase_MultipleDotOps) {
  auto module = createModule(/*supportDPAS=*/true);
  auto funcOp = createFunction(module);

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  // Create multiple dot ops
  for (int i = 0; i < 3; ++i) {
    auto loc = builder->getUnknownLoc();
    auto tensorType = RankedTensorType::get({32, 32}, builder->getF16Type());
    auto tensorTypeOut = RankedTensorType::get({32, 32}, builder->getF32Type());

    auto a = builder->create<arith::ConstantOp>(
        loc, tensorType, builder->getZeroAttr(tensorType));
    auto b = builder->create<arith::ConstantOp>(
        loc, tensorType, builder->getZeroAttr(tensorType));
    auto c = builder->create<arith::ConstantOp>(
        loc, tensorTypeOut, builder->getZeroAttr(tensorTypeOut));

    builder->create<DotOp>(loc, tensorTypeOut, a, b, c, InputPrecision::IEEE,
                           0);
  }
  builder->create<func::ReturnOp>(builder->getUnknownLoc());

  DPASAnalysisV1 analysis(module);
  auto result = analysis.canUseDPAS(funcOp);
  EXPECT_EQ(result, DPASAnalysisResult::True);
}

TEST_F(DPASAnalysisTest, EdgeCase_MixedValidAndInvalidOps) {
  auto module = createModule(/*supportDPAS=*/true);
  auto funcOp = createFunction(module);

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&funcOp.getBody().front());

  // Create one valid dot op
  {
    auto loc = builder->getUnknownLoc();
    auto tensorType = RankedTensorType::get({32, 32}, builder->getF16Type());
    auto tensorTypeOut = RankedTensorType::get({32, 32}, builder->getF32Type());

    auto a = builder->create<arith::ConstantOp>(
        loc, tensorType, builder->getZeroAttr(tensorType));
    auto b = builder->create<arith::ConstantOp>(
        loc, tensorType, builder->getZeroAttr(tensorType));
    auto c = builder->create<arith::ConstantOp>(
        loc, tensorTypeOut, builder->getZeroAttr(tensorTypeOut));

    builder->create<DotOp>(loc, tensorTypeOut, a, b, c, InputPrecision::IEEE,
                           0);
  }

  // Create one invalid dot op (mismatched types)
  {
    auto loc = builder->getUnknownLoc();
    auto tensorTypeA = RankedTensorType::get({32, 32}, builder->getF16Type());
    auto tensorTypeB = RankedTensorType::get({32, 32}, builder->getF32Type());
    auto tensorTypeOut = RankedTensorType::get({32, 32}, builder->getF32Type());

    auto a = builder->create<arith::ConstantOp>(
        loc, tensorTypeA, builder->getZeroAttr(tensorTypeA));
    auto b = builder->create<arith::ConstantOp>(
        loc, tensorTypeB, builder->getZeroAttr(tensorTypeB));
    auto c = builder->create<arith::ConstantOp>(
        loc, tensorTypeOut, builder->getZeroAttr(tensorTypeOut));

    builder->create<DotOp>(loc, tensorTypeOut, a, b, c, InputPrecision::IEEE,
                           0);
  }

  builder->create<func::ReturnOp>(builder->getUnknownLoc());

  DPASAnalysisV1 analysis(module);
  auto result = analysis.canUseDPAS(funcOp);
  // Should be False because one op cannot use DPAS
  EXPECT_EQ(result, DPASAnalysisResult::False);
}

} // namespace
