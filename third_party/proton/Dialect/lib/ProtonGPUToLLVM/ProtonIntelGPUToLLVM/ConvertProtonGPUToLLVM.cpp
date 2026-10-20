#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/IntelPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton::gpu {
#define GEN_PASS_DEF_CONVERTPROTONINTELGPUTOLLVM
#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/Passes.h.inc"
} // namespace triton::proton::gpu
} // namespace mlir

namespace {

class ProtonLLVMConversionTarget : public ConversionTarget {
public:
  explicit ProtonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<mlir::triton::proton::gpu::ProtonGPUDialect>();
    addIllegalDialect<mlir::triton::proton::ProtonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertProtonIntelGPUToLLVM
    : public mlir::triton::proton::gpu::impl::ConvertProtonIntelGPUToLLVMBase<
          ConvertProtonIntelGPUToLLVM> {
  explicit ConvertProtonIntelGPUToLLVM() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    std::unique_ptr<mlir::triton::intel::TargetInfo> tritonTargetInfo =
        mlir::triton::intel::createTargetInfo(mod);
    auto protonTargetInfo =
        mlir::triton::proton::gpu::Intel::TargetInfo(*tritonTargetInfo);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               *tritonTargetInfo);
    populateTypeConversions(typeConverter, protonTargetInfo);
    mlir::triton::proton::gpu::populateProtonGPUOpPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::triton::proton::gpu::Intel::populateProtonGPUOpIntelPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToLLVMSPVConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    auto convTarget = ProtonLLVMConversionTarget(*context);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns),
                                      config)))
      return signalPassFailure();

    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonIntelGPUToLLVMPass() {
  return std::make_unique<ConvertProtonIntelGPUToLLVM>();
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
