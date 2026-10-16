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

    // `populateProtonGPUOpPatterns` fabricates fresh `gpu.block_id` /
    // `gpu.grid_dim` ops (see `getLinearId()`) when lowering
    // `proton_gpu.init`. By this point in the Intel pipeline the main
    // `add_to_llvmir`/`add_gen_to_llvm` passes have already run, so nothing
    // downstream will lower these -- pull in the same upstream MLIR
    // conversion the main Intel pipeline uses for `tt.get_program_id`
    // (`PipelineManager::populateConversionPatterns`) so the fixpoint-driven
    // partial conversion below also cascades through them: it lowers
    // `gpu.block_id`/`gpu.grid_dim`/etc. straight to calls to the SPIR-V
    // OpenCL builtins (`get_group_id`, ...), no TritonGEN hop needed.
    mlir::populateGpuToLLVMSPVConversionPatterns(typeConverter, patterns);

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);

    auto convTarget = ProtonLLVMConversionTarget(*context);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns),
                                      config)))
      return signalPassFailure();

    // The various type conversions above (segment/memdesc/pointer) can leave
    // behind unrealized_conversion_cast pairs that LLVM translation can't
    // handle; fold them away.
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
