// Pins `getGRFBytesPerHardwareThread`'s budget table: every GRF mode times the
// register width from `ttig.grf_register_size`, with "default" and "auto" at
// the 128-register budget and an absent or invalid width at 32 B.
#include "intel/include/Analysis/RegisterPressure.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {

class RegisterPressureGRFModeTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  // `registerSize` is left unset when absent (the hand-written-IR case).
  OwningOpRef<ModuleOp> createModule(std::optional<int> registerSize,
                                     int threadsPerWarp = 16) {
    auto loc = builder->getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    (*module)->setAttr(AttrNumWarpsName, builder->getI32IntegerAttr(4));
    (*module)->setAttr(AttrNumThreadsPerWarp,
                       builder->getI32IntegerAttr(threadsPerWarp));
    if (registerSize)
      (*module)->setAttr(TritonIntelGPUDialect::getGRFRegisterSizeAttrName(),
                         builder->getI32IntegerAttr(*registerSize));
    return module;
  }

  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

unsigned budgetBytes(ModuleOp mod, StringRef grfMode) {
  return RegisterPressureAnalysis::getGRFBytesPerHardwareThread(grfMode, mod);
}

TEST_F(RegisterPressureGRFModeTest, ExplicitModes64ByteRegisters) {
  auto module = createModule(64);
  EXPECT_EQ(budgetBytes(*module, "128"), 8192u);
  EXPECT_EQ(budgetBytes(*module, "160"), 10240u);
  EXPECT_EQ(budgetBytes(*module, "192"), 12288u);
  EXPECT_EQ(budgetBytes(*module, "256"), 16384u);
  EXPECT_EQ(budgetBytes(*module, "512"), 32768u);
}

TEST_F(RegisterPressureGRFModeTest, ExplicitModes32ByteRegisters) {
  auto module = createModule(32);
  EXPECT_EQ(budgetBytes(*module, "128"), 4096u);
  EXPECT_EQ(budgetBytes(*module, "256"), 8192u);
}

TEST_F(RegisterPressureGRFModeTest, DefaultAndAutoUse128Registers) {
  // A "default" or "auto" compile starts at 128 registers.
  auto module = createModule(64);
  EXPECT_EQ(budgetBytes(*module, "default"), 8192u);
  EXPECT_EQ(budgetBytes(*module, "auto"), 8192u);
}

TEST_F(RegisterPressureGRFModeTest, AbsentRegisterSizeFallsBackTo32Bytes) {
  auto module = createModule(std::nullopt);
  EXPECT_EQ(budgetBytes(*module, "default"), 4096u);
  EXPECT_EQ(budgetBytes(*module, "256"), 8192u);
}

TEST_F(RegisterPressureGRFModeTest, InvalidRegisterSizeFallsBackTo32Bytes) {
  // Only 32 and 64 are known widths; anything else is treated as absent.
  auto module = createModule(48);
  EXPECT_EQ(budgetBytes(*module, "default"), 4096u);
}

TEST_F(RegisterPressureGRFModeTest, PerLaneBudgetDividesByThreadsPerWarp) {
  auto module16 = createModule(64, /*threadsPerWarp=*/16);
  EXPECT_EQ(RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes("default",
                                                                 *module16),
            512u);
  auto module32 = createModule(64, /*threadsPerWarp=*/32);
  EXPECT_EQ(RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes("default",
                                                                 *module32),
            256u);
}

TEST_F(RegisterPressureGRFModeTest, MaxGRFModeDoesNotRaiseDefaultBudget) {
  // `ttig.max_grf_mode` bounds AccelerateMatmul only; "default" and "auto"
  // stay at the 128-register budget whatever it says.
  auto module = createModule(std::nullopt);
  (*module)->setAttr(TritonIntelGPUDialect::getMaxGRFModeAttrName(),
                     builder->getStringAttr("512"));
  EXPECT_EQ(budgetBytes(*module, "default"), 4096u);
  EXPECT_EQ(budgetBytes(*module, "auto"), 4096u);
}

#ifndef NDEBUG
// The `assert()` this pins is compiled out under NDEBUG (e.g. a Release
// build), so there is nothing for EXPECT_DEATH to observe there.
TEST_F(RegisterPressureGRFModeTest, UnrecognizedGRFModeAsserts) {
  // Every caller narrows grfMode to an explicit mode, "default" or "auto".
  auto module = createModule(std::nullopt);
  EXPECT_DEATH(budgetBytes(*module, "not-a-real-mode"),
               "grfMode must be an explicit mode");
}
#endif // NDEBUG

} // namespace
