// Pins `getGRFBytesPerHardwareThread`'s `UnknownGRFSizeAssumption::Largest`
// resolution table end to end: every `ttig.max_grf_mode` value the Python
// producer (`get_max_grf_mode` in compiler.py) can stamp, the values it
// deliberately does not, and the `num_warps > 32` exception (issue #8074,
// review findings 1 and 6). A lit test exercising the same table through
// `ReduceVariableLiveness`'s sink decision cannot distinguish an explicit
// "512" from an absent attribute (both resolve to the same 16384-byte
// budget), so this is checked directly here instead.
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
  // "128" is not a value the Python producer ever stamps (see #8074's
  // review, finding 3): a kernel already gets 128-GRF without any
  // escalation, so a *maximum auto-escalation target* of "128" is
  // incoherent. Documented here rather than silently assumed: if this ever
  // starts mattering (a hand-written module or a future producer stamping
  // it), `Largest` degenerates to exactly `Smallest`'s answer.
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
  // `make_zebin`'s automatic-escalation retry -- the only path that would
  // ever realize `ttig.max_grf_mode` on the `grf_mode='default'` path -- is
  // itself skipped outright once `num_warps > 32`, so the realizable
  // ceiling is `Smallest`'s answer regardless of what the attribute says.
  auto module = createModule(StringRef("512"), /*numWarps=*/64);
  EXPECT_EQ(largestBytes(*module, "default"), 4096u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsOver32DoesNotCapAutoMode) {
  // `grf_mode='auto'` escalates inside IGC, not through `make_zebin`'s
  // retry, and is not itself gated on `num_warps` at the backend level, so
  // the `num_warps > 32` exception must not apply to it.
  auto module = createModule(StringRef("512"), /*numWarps=*/64);
  EXPECT_EQ(largestBytes(*module, "auto"), 16384u);
}

TEST_F(RegisterPressureGRFModeTest, NumWarpsAtBoundaryIsUnaffected) {
  // 32 itself is still eligible for escalation (`make_zebin` guards on
  // `num_warps <= 32`), so the cap must not fire here.
  auto module = createModule(StringRef("512"), /*numWarps=*/32);
  EXPECT_EQ(largestBytes(*module, "default"), 16384u);
}

} // namespace
