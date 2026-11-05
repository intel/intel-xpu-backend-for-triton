// Pins `getGRFBytesPerHardwareThread`'s `UnknownGRFSizeAssumption::Largest`
// resolution table end to end: every `ttig.max_grf_mode` value the Python
// producer (`get_max_grf_mode` in compiler.py) can stamp, the values it
// deliberately does not, and the `num_warps > 32` exception. A lit test
// exercising the same table through `ReduceVariableLiveness`'s sink decision
// cannot distinguish an explicit "512" from an absent attribute (both
// resolve to the same 16384-byte budget), so this is checked directly here
// instead.
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

  // `maxGRFMode` is left unset when absent (the common hand-written-IR case
  // `RegisterPressureAnalysis`'s own doc describes).
  OwningOpRef<ModuleOp> createModule(std::optional<StringRef> maxGRFMode,
                                     int numWarps = 4) {
    auto loc = builder->getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    (*module)->setAttr(AttrNumWarpsName, builder->getI32IntegerAttr(numWarps));
    (*module)->setAttr(AttrNumThreadsPerWarp, builder->getI32IntegerAttr(16));
    if (maxGRFMode)
      (*module)->setAttr(TritonIntelGPUDialect::getMaxGRFModeAttrName(),
                         builder->getStringAttr(*maxGRFMode));
    return module;
  }

  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

unsigned largestBytes(ModuleOp mod, StringRef grfMode = "default") {
  return RegisterPressureAnalysis::getGRFBytesPerHardwareThread(
      grfMode, mod,
      RegisterPressureAnalysis::UnknownGRFSizeAssumption::Largest);
}

TEST_F(RegisterPressureGRFModeTest, AbsentAttributeFallsBackTo512Mode) {
  auto module = createModule(std::nullopt);
  EXPECT_EQ(largestBytes(*module), 16384u);
}

TEST_F(RegisterPressureGRFModeTest, ExplicitAttr256) {
  auto module = createModule(StringRef("256"));
  EXPECT_EQ(largestBytes(*module), 8192u);
}

TEST_F(RegisterPressureGRFModeTest, ExplicitAttr512) {
  auto module = createModule(StringRef("512"));
  EXPECT_EQ(largestBytes(*module), 16384u);
}

TEST_F(RegisterPressureGRFModeTest,
       ExplicitAttr128CollapsesLargestIntoSmallest) {
  // "128" is not a value the built-in "cri"-vs-everything-else policy ever
  // stamps automatically (a kernel already gets 128-GRF without any
  // escalation, so a *maximum auto-escalation target* of "128" is
  // incoherent), though a driver- or out-of-tree-supplied override can still
  // set it explicitly. Documented here rather than silently assumed: however
  // it gets there, `Largest` degenerates to exactly `Smallest`'s answer.
  auto module = createModule(StringRef("128"));
  EXPECT_EQ(largestBytes(*module), 4096u);
}

TEST_F(RegisterPressureGRFModeTest, InvalidAttrFallsThroughToAbsenceDefault) {
  // A typo or a future mode this table doesn't know: falls through to the
  // same behaviour-preserving default as no attribute at all, rather than
  // silently resolving to the wrong budget.
  auto module = createModule(StringRef("bogus"));
  EXPECT_EQ(largestBytes(*module), 16384u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsOver32CapsDefaultModeAtSmallest) {
  // A larger GRF mode halves (256) or quarters (512) the maximum launchable
  // work-group size, so a `num_warps > 32` kernel can never actually run at
  // a larger mode, regardless of what the attribute says.
  auto module = createModule(StringRef("512"), /*numWarps=*/64);
  EXPECT_EQ(largestBytes(*module, "default"), 4096u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsOver32CapsAutoModeAtSmallestToo) {
  // The same cap applies regardless of which mechanism would have picked
  // the larger mode: `'auto'` is capped here just like `'default'` is.
  auto module = createModule(StringRef("256"), /*numWarps=*/64);
  EXPECT_EQ(largestBytes(*module, "auto"), 4096u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsAtBoundaryIsUnaffected) {
  // 32 itself is still launchable at a larger GRF mode, so the cap must not
  // fire here.
  auto module = createModule(StringRef("512"), /*numWarps=*/32);
  EXPECT_EQ(largestBytes(*module, "default"), 16384u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsAtBoundaryIsUnaffectedForAutoToo) {
  auto module = createModule(StringRef("512"), /*numWarps=*/32);
  EXPECT_EQ(largestBytes(*module, "auto"), 16384u);
}

TEST_F(RegisterPressureGRFModeTest, AutoModeRespectsMaxGRFMode) {
  // 'auto' resolves `ttig.max_grf_mode` the same way 'default' does. A
  // target whose max_grf_mode is "256" (non-"cri") collapses 'auto' to the
  // same 8192-byte budget as 'default', not the unconditional 16384-byte
  // bound an absent attribute falls back to.
  auto module = createModule(StringRef("256"));
  EXPECT_EQ(largestBytes(*module, "auto"), 8192u);
}

TEST_F(RegisterPressureGRFModeTest, ExplicitGRFMode160And192) {
  // "160" and "192" (accepted on "cri" only) are explicit modes like the
  // others: their exact budget, regardless of `ttig.max_grf_mode`.
  auto module = createModule(StringRef("512"));
  EXPECT_EQ(largestBytes(*module, "160"), 5120u);
  EXPECT_EQ(largestBytes(*module, "192"), 6144u);
}

#ifndef NDEBUG
// The `assert()` this pins is compiled out under NDEBUG (e.g. a Release
// build), so there is nothing for EXPECT_DEATH to observe there; guard the
// whole test rather than let it fail on a build where the invariant it
// checks cannot fire.
TEST_F(RegisterPressureGRFModeTest, UnrecognizedGRFModeAsserts) {
  // A typo'd or otherwise-unrecognized grf-mode string (neither an explicit
  // mode, "default", nor "auto") is an internal-invariant violation, not
  // input this function is expected to recover from: every caller is
  // supposed to have already narrowed grfMode to one of those five values.
  auto module = createModule(/*maxGRFMode=*/std::nullopt);
  EXPECT_DEATH(largestBytes(*module, "not-a-real-mode"),
               "grfMode must be an explicit mode");
}
#endif // NDEBUG

} // namespace
