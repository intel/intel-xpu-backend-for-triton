#include "PipelineManager.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_CONVERTTRITONINTELGPUTOLLVM
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addIllegalDialect<triton::TritonGEN::TritonGENDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::gpu::intel::TritonIntelGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addDynamicallyLegalOp<ModuleOp>(
        [](ModuleOp op) { return spirv::lookupTargetEnv(op) != nullptr; });
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::gpu::intel::impl::ConvertTritonIntelGPUToLLVMBase<
          ConvertTritonGPUToLLVM> {
  using ConvertTritonIntelGPUToLLVMBase::ConvertTritonIntelGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, TritonGEN::TritonGENDialect,
                    spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    intel::TritonGPUToLLVMPipelineManager pipelineManager(mod, context);
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    bool isExperimentalPathEnabled =
        mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                         getSupportSG2DBlockAttrName()) &&
        mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                         getSupportDPASAttrName()) &&
        mlir::triton::tools::getBoolEnv("TRITON_INTEL_ENABLE_BLOCK_PTR");
    TritonIntelGPUToLLVMTypeConverter typeConverter(context, option,
                                                    isExperimentalPathEnabled);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    if (!pipelineManager.skipSharedMemoryAllocation()) {
      ModuleAllocation allocation(mod);
      ModuleMembarAnalysis membarPass(&allocation);
      membarPass.run();
    }

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonIntelGPUToLLVMTypeConverter typeConverter(
          context, option, isExperimentalPathEnabled);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      pipelineManager.populateFunctionConversionPatterns(
          funcPatterns, typeConverter, numWarps);

      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    mlir::triton::intel::TargetInfo targetInfo;
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    pipelineManager.populateConversionPatterns(
        patterns, axisInfoAnalysis, typeConverter, targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i)
        funcOp.removeArgAttr(i, "tt.divisibility");
    });
  }
};

} // anonymous namespace
